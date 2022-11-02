########################################################################################
#
# M5 model as gender classifier.
#
# Author(s): Nik Vaessen
########################################################################################

from dataclasses import dataclass
from typing import Callable, List

import torch as t

import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from src.networks.gr_module import GenderRecognitionLightningModule

########################################################################################
# M5 as nn module


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        return x


########################################################################################
# Wrap M5 as GenderRecognitionModule


@dataclass
class M5GenderClassifierConfig:
    n_input: int
    n_output: int
    stride: int
    n_channel: int


class M5GenderClassifier(GenderRecognitionLightningModule):
    def __init__(
        self,
        cfg: M5GenderClassifierConfig,
        hyperparameter_config: DictConfig,
        loss_fn_constructor: Callable[[], Callable[[t.Tensor, t.Tensor], t.Tensor]],
        test_names: List[str],
    ):
        super().__init__(hyperparameter_config, loss_fn_constructor, test_names)

        self.cfg = cfg
        self.m5 = M5(
            n_input=self.cfg.n_input,
            n_output=self.cfg.n_output,
            stride=self.cfg.stride,
            n_channel=self.cfg.n_channel,
        )

    def forward(self, network_input: t.Tensor) -> t.Tensor:
        assert len(network_input.shape) == 2
        network_input = t.unsqueeze(network_input, 1)

        classification = self.m5(network_input)

        return t.squeeze(classification)
