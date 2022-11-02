########################################################################################
#
# WavLM as gender classifier
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List

import torch as t

from omegaconf import DictConfig
from transformers.models.wavlm import WavLMModel

from src.networks.gr_module import GenderRecognitionLightningModule

########################################################################################
# config
from src.networks.xvector import GenderClassifier
from src.util.freeze import FreezeManager


@dataclass
class WavLMGenderClassifierConfig:
    # model to load for feature extraction
    huggingface_id: str
    embedding_size: int

    # freeze logic
    freeze_cnn: bool
    freeze_transformer: bool
    num_steps_freeze_cnn: int
    num_steps_freeze_transformer: int

    # classifier shape
    num_fc1: int
    num_fc2: int

    # dropout
    drop_prob: float


########################################################################################
# network implementation


class WavLMGenderClassifier(GenderRecognitionLightningModule):
    def __init__(
        self,
        cfg: WavLMGenderClassifierConfig,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        test_names: List[str],
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor, test_names)

        self.cfg = cfg
        self.wavLM: WavLMModel = WavLMModel.from_pretrained(self.cfg.huggingface_id)
        self.wavLM.feature_extractor.requires_grad_(False)

        self.classifier = GenderClassifier(
            in_dim=self.cfg.embedding_size,
            fc1_nodes=self.cfg.num_fc1,
            fc2_nodes=self.cfg.num_fc2,
            dropout_prob=self.cfg.drop_prob,
        )

        # freeze logic
        self.freeze_cnn = FreezeManager(
            module=self.wavLM.feature_extractor,
            is_frozen_at_init=self.cfg.freeze_cnn,
            num_steps_frozen=self.cfg.num_steps_freeze_cnn,
        )
        self.freeze_transformer = FreezeManager(
            module=[
                x
                for x in (
                    self.wavLM.feature_projection,
                    self.wavLM.encoder,
                    self.wavLM.masked_spec_embed,
                    self.wavLM.adapter,
                )
                if x is not None
            ],
            is_frozen_at_init=self.cfg.freeze_transformer,
            num_steps_frozen=self.cfg.num_steps_freeze_transformer,
        )

    def forward(self, network_input: t.Tensor) -> t.Tensor:
        if len(network_input.shape) == 3 and network_input.shape[1] == 1:
            network_input = t.squeeze(network_input)
            network_input = t.unsqueeze(network_input, 0)

        embedding_sequence = self.wavLM(network_input).last_hidden_state

        # pool sequence into fixed-size embedding
        embedding = t.mean(embedding_sequence, dim=1)

        classification = self.classifier(embedding)

        return classification

    def on_train_start(self) -> None:
        self.freeze_cnn.on_train_start()
        self.freeze_transformer.on_train_start()

    def on_after_backward(self) -> None:
        self.freeze_cnn.on_after_backward()
        self.freeze_transformer.on_after_backward()
