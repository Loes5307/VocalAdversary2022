########################################################################################
#
# This file implement a datamodule for the LibriSpeech dataset.
#
# Author(s): Nik Vaessen
########################################################################################

import json
import pathlib

from dataclasses import dataclass
from typing import Optional, List

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from data_utility.eval.speaker.evaluator import SpeakerTrial
from data_utility.pipe.builder import GenderRecognitionDataPipeBuilder
from src.util.config_util import CastingConfig


########################################################################################
# config


@dataclass
class GenderRecognitionDataModuleConfig(CastingConfig):
    # name of train dataset (for log)
    name: str

    # path to folder(s) containing train data
    train_shard_paths: List[pathlib.Path]

    # path to folder(s) containing val data
    val_shard_paths: List[pathlib.Path]

    # shard pattern
    shard_file_pattern: str

    # name of each test set
    test_names: List[str]

    # path to each shard of test set (only 1 dir each)
    test_shards: List[pathlib.Path]

    # path to each trial list matching the test set
    test_trials: List[pathlib.Path]

    # define labels
    female_id: str = "f"
    female_idx: int = 0
    male_id: str = "m"
    male_idx: int = 1


########################################################################################
# implementation


class GenderRecognitionDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: GenderRecognitionDataModuleConfig,
        train_pipe_builder: GenderRecognitionDataPipeBuilder,
        val_pipe_builder: GenderRecognitionDataPipeBuilder,
        test_pipe_builder: GenderRecognitionDataPipeBuilder,
    ):
        super(GenderRecognitionDataModule, self).__init__()

        self.cfg = cfg

        self.train_pipe_builder = train_pipe_builder
        self.val_pipe_builder = val_pipe_builder
        self.test_pipe_builder = test_pipe_builder
        self._set_gender_to_idx()

        if not (
            len(self.cfg.test_names)
            == len(self.cfg.test_shards)
            == len(self.cfg.test_trials)
        ):
            raise ValueError("length of test names, shards, and trials does not match")

        # init in setup()
        self.train_dp = None
        self.val_dp = None
        self.test_dp_list = None

    def _set_gender_to_idx(self):
        for dpb in [
            self.train_pipe_builder,
            self.val_pipe_builder,
            self.test_pipe_builder,
        ]:
            dpb.set_gender_idx(
                female_str=self.cfg.female_id,
                female_idx=self.cfg.female_idx,
                male_str=self.cfg.male_id,
                male_idx=self.cfg.male_idx,
            )

    def get_num_genders(self) -> int:
        return 2

    def get_test_names(self):
        return self.cfg.test_names

    def setup(self, stage: Optional[str] = None) -> None:
        # train dp
        self.train_dp = self.train_pipe_builder.get_pipe(
            shard_dirs=self.cfg.train_shard_paths,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )

        # val dp
        self.val_dp = self.val_pipe_builder.get_pipe(
            shard_dirs=self.cfg.val_shard_paths,
            shard_file_pattern=self.cfg.shard_file_pattern,
        )

        # test dp
        self.test_dp_list = [
            self.test_pipe_builder.get_pipe(
                shard_dirs=path, shard_file_pattern=self.cfg.shard_file_pattern
            )
            for path in self.cfg.test_shards
        ]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_pipe_builder.wrap_pipe(self.train_dp)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_pipe_builder.wrap_pipe(self.val_dp)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [self.test_pipe_builder.wrap_pipe(dp) for dp in self.test_dp_list]
