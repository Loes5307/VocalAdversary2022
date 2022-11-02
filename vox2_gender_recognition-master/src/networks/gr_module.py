################################################################################
#
# Define a base lightning module for a gender recognition network.
#
# Author(s): Nik Vaessen
################################################################################

import logging

from abc import abstractmethod
from typing import Callable, Optional, List, Union

import torch as t
import torchmetrics as tm

import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from data_utility.pipe.containers import GenderRecognitionBatch

################################################################################
# Definition of speaker recognition API

# A logger for this file

log = logging.getLogger(__name__)


class GenderRecognitionLightningModule(pl.LightningModule):
    def __init__(
        self,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        test_names: List[str],
    ):
        super().__init__()

        # input arguments
        self.loss_fn = loss_fn_constructor()

        # created by set_methods
        self.optimizer = None
        self.schedule = None

        # log hyperparameters
        self.save_hyperparameters(OmegaConf.to_container(hyperparameter_config))

        # metric loggers
        self.metric_train_acc = tm.Accuracy()
        self.metric_train_loss = tm.MeanMetric()
        self.metric_val_acc = tm.Accuracy()
        self.metric_val_loss = tm.MeanMetric()

        # names for each test dataloader idx
        self.test_names = test_names

    @abstractmethod
    def forward(self, network_input: t.Tensor) -> t.Tensor:
        # return a tensor of shape [BS, 2] where
        # idx 0 = P(network_input=female), and idx 1 = P(network_input=male)
        pass

    def training_step(
        self,
        batch: GenderRecognitionBatch,
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> STEP_OUTPUT:
        assert isinstance(batch, GenderRecognitionBatch)

        classification = self(batch.audio_tensor)
        loss, probabilities = self.loss_fn(classification, batch.id_tensor)

        # log metrics
        self.metric_train_acc(probabilities, batch.id_tensor)
        self.metric_train_loss(loss)

        if batch_idx % 100 == 0:
            self.log("train_acc", self.metric_train_acc, prog_bar=True)
            self.log("train_loss", self.metric_train_loss, prog_bar=False)
            self.metric_train_acc.reset()
            self.metric_train_loss.reset()

        return loss

    def validation_step(
        self,
        batch: GenderRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        assert isinstance(batch, GenderRecognitionBatch)

        classification = self(batch.audio_tensor)
        loss, probabilities = self.loss_fn(classification, batch.id_tensor)

        # log metrics
        self.metric_val_acc(probabilities, batch.id_tensor)
        self.metric_val_loss(loss)

        return loss

    def validation_epoch_end(
        self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]
    ) -> None:
        self.log("val_acc", self.metric_val_acc, prog_bar=True)
        self.log("val_loss", self.metric_val_loss, prog_bar=True)

    def test_step(
        self,
        batch: GenderRecognitionBatch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Optional[STEP_OUTPUT]:
        classification = self(batch.audio_tensor)

        if len(classification.shape) == 1:
            classification = t.unsqueeze(classification, 0)

        _, probabilities = self.loss_fn(classification, batch.id_tensor)

        return {
            "prediction": probabilities.cpu(),
            "ground_truth": batch.id_tensor.cpu(),
        }

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        if len(self.test_names) == 1:
            outputs = [outputs]

        for dl_idx, output in enumerate(outputs):
            test_name = self.test_names[dl_idx]

            acc_metric = tm.Accuracy()

            for output_dict in output:
                acc_metric(output_dict["prediction"], output_dict["ground_truth"])

            self.log(f"test_{test_name}_acc", acc_metric.compute())

    def set_optimizer(self, optimizer: t.optim.Optimizer):
        self.optimizer = optimizer

    def set_lr_schedule(self, schedule: t.optim.lr_scheduler._LRScheduler):
        self.schedule = schedule

    def configure_optimizers(self):
        if self.optimizer is None:
            raise ValueError("optimizer not set")
        if self.schedule is None:
            raise ValueError("LR schedule not set")

        return [self.optimizer], [self.schedule]
