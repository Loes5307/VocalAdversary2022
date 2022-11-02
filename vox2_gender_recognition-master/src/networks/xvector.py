########################################################################################
#
# X-Vector as gender classifier.
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List

import torch as t
import torchaudio

from omegaconf import DictConfig

from speechbrain.lobes.models.Xvector import Xvector

from src.networks.gr_module import GenderRecognitionLightningModule

########################################################################################
# config


@dataclass
class XVectorGenderClassifierConfig:
    # feature extractor shape
    num_in_channels: int
    num_out_channels: int

    # classifier shape
    num_fc1: int
    num_fc2: int

    # dropout
    drop_prob: float


########################################################################################
# network implementation


class GenderClassifier(t.nn.Module):
    def __init__(
        self, in_dim: int, fc1_nodes: int, fc2_nodes: int, dropout_prob: float
    ):
        super().__init__()

        self.fc1 = self._block(in_dim, fc1_nodes, dropout_prob)
        self.fc2 = self._block(fc1_nodes, fc2_nodes, dropout_prob)
        self.classification_layer = t.nn.Linear(in_features=fc2_nodes, out_features=2)

    def forward(self, x: t.Tensor):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classification_layer(x)

        return x

    @staticmethod
    def _block(in_dim: int, out_dim: int, dropout_prob: float):
        return t.nn.Sequential(
            t.nn.Linear(in_features=in_dim, out_features=out_dim),
            t.nn.LeakyReLU(),
            t.nn.Dropout(p=dropout_prob),
        )


class XVectorGenderClassifier(GenderRecognitionLightningModule):
    def __init__(
        self,
        cfg: XVectorGenderClassifierConfig,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        test_names: List[str],
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor, test_names)

        self.cfg = cfg

        self.mfcc = torchaudio.transforms.MFCC(n_mfcc=self.cfg.num_in_channels)
        self.feature_extractor = Xvector(
            in_channels=cfg.num_in_channels, lin_neurons=self.cfg.num_out_channels
        )
        self.classifier = GenderClassifier(
            self.cfg.num_out_channels,
            self.cfg.num_fc1,
            self.cfg.num_fc2,
            self.cfg.drop_prob,
        )

    def forward(self, network_input: t.Tensor) -> t.Tensor:
        assert len(network_input.shape) == 2

        # first calculate MFCC and transpose time/feature axis
        network_input = self.mfcc(network_input)
        network_input = network_input.transpose(1, 2)

        # calculate embedding
        embedding = self.feature_extractor(network_input)
        embedding = embedding.squeeze()

        # calculate prediction
        classification = self.classifier(embedding)

        return classification
