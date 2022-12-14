########################################################################################
#
# This file is the main entrypoint of the train/eval loop based on the
# hydra configuration.
#
# Author(s): Nik Vaessen
########################################################################################

import logging

from typing import List, Dict

import torch as t
import pytorch_lightning as pl
import transformers
import wandb

from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import instantiate
from torch.distributed import destroy_process_group
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_utility.pipe.builder import GenderRecognitionDataPipeBuilder
from src.data.module import (
    GenderRecognitionDataModuleConfig,
    GenderRecognitionDataModule,
)
from src.networks.gr_module import GenderRecognitionLightningModule
from src.networks.m5 import M5GenderClassifierConfig, M5GenderClassifier
from src.networks.wavLM import WavLMGenderClassifierConfig, WavLMGenderClassifier
from src.networks.xvector import XVectorGenderClassifierConfig, XVectorGenderClassifier
from src.util.system import get_git_revision_hash

log = logging.getLogger(__name__)

########################################################################################
# implement constructing data module


def construct_gender_recognition_data_pipe_builders(cfg: DictConfig):
    train_dp = GenderRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.gender_datapipe.train_dp)
    )
    val_dp = GenderRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.gender_datapipe.val_dp)
    )
    test_dp = GenderRecognitionDataPipeBuilder(
        cfg=instantiate(cfg.data.gender_datapipe.test_dp)
    )

    return train_dp, val_dp, test_dp


def construct_data_module(cfg: DictConfig):
    dm_cfg = instantiate(cfg.data.module)

    if isinstance(dm_cfg, GenderRecognitionDataModuleConfig):
        train_dpb, val_dpb, test_dpb = construct_gender_recognition_data_pipe_builders(
            cfg
        )

        dm = GenderRecognitionDataModule(
            dm_cfg,
            train_pipe_builder=train_dpb,
            val_pipe_builder=val_dpb,
            test_pipe_builder=test_dpb,
        )
    else:
        raise ValueError(f"no suitable constructor for {dm_cfg}")

    return dm


########################################################################################
# implement the construction of network modules


def init_model(cfg: DictConfig, network_class, kwargs: Dict):
    # load model weights from checkpoint
    potential_checkpoint_path = cfg.get("load_network_from_checkpoint", None)

    if potential_checkpoint_path is not None:
        log.info(
            f"reloading {network_class.__class__} from {potential_checkpoint_path}"
        )
        network = network_class.load_from_checkpoint(
            cfg.load_network_from_checkpoint, strict=False, **kwargs
        )
    else:
        network = network_class(**kwargs)

    return network


def construct_network_module(
    cfg: DictConfig, test_names: List[str]
) -> GenderRecognitionLightningModule:
    # load loss function
    def loss_fn_constructor():
        # should be instantiated in the network
        # so that potential parameters are properly
        # registered
        return instantiate(cfg.optim.loss)

    # load network config
    network_cfg = instantiate(cfg.network)

    if isinstance(network_cfg, XVectorGenderClassifierConfig):
        network_clazz = XVectorGenderClassifier
    elif isinstance(network_cfg, WavLMGenderClassifierConfig):
        network_clazz = WavLMGenderClassifier
    elif isinstance(network_cfg, M5GenderClassifierConfig):
        network_clazz = M5GenderClassifier
    else:
        raise ValueError(f"unknown {network_cfg=}")

    kwargs = {
        "loss_fn_constructor": loss_fn_constructor,
        "hyperparameter_config": cfg,
        "cfg": network_cfg,
        "test_names": test_names,
    }
    network = init_model(cfg, network_clazz, kwargs)

    # set optimizer and learning rate schedule
    optimizer = instantiate(cfg.optim.algo, params=network.parameters())
    schedule = {
        "scheduler": instantiate(cfg.optim.schedule.scheduler, optimizer=optimizer),
        "monitor": cfg.optim.schedule.monitor,
        "interval": cfg.optim.schedule.interval,
        "frequency": cfg.optim.schedule.frequency,
        "name": cfg.optim.schedule.name,
    }
    # remove None values from dict
    schedule = {k: v for k, v in schedule.items() if v is not None}

    network.set_optimizer(optimizer)
    network.set_lr_schedule(schedule)

    return network


########################################################################################
# implement construction of callbacks, profiler and logger


def construct_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks = []

    callback_cfg: DictConfig = cfg.callbacks

    ModelCheckpoint.CHECKPOINT_NAME_LAST = callback_cfg.get(
        "last_checkpoint_pattern", "last"
    )

    for cb_key in callback_cfg.to_add:
        if cb_key is None:
            continue

        if cb_key in callback_cfg:
            cb = instantiate(callback_cfg[cb_key])
            log.info(f"Using callback <{cb}>")

            callbacks.append(instantiate(callback_cfg[cb_key]))

    return callbacks


def construct_profiler(cfg: DictConfig):
    profile_cfg = cfg.get("profiler", None)

    if profile_cfg is None:
        return None
    else:
        return instantiate(profile_cfg)


def construct_logger(cfg: DictConfig):
    if cfg.use_wandb:
        tag = [cfg.date_tag]

        if isinstance(cfg.tag, str):
            tag.append(cfg.tag)

        if isinstance(cfg.tag, ListConfig):
            tag.extend(cfg.tag)

        logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.experiment_name,
            tags=tag,
        )
        # init the wandb agent
        _ = logger.experiment
    else:
        logger = True

    return logger


########################################################################################
# implement the main function based on the whole config


def run_train_eval_script(cfg: DictConfig):
    # create logger
    logger = construct_logger(cfg)

    # set random seed for main script and workers
    pl.seed_everything(cfg.seed, workers=True)

    # print config
    print(OmegaConf.to_yaml(cfg))
    print(f"current git commit hash: {get_git_revision_hash()}")
    print(f"PyTorch version is {t.__version__}")
    print(f"PyTorch Lightning version is {pl.__version__}")
    print(f"transformers version is {transformers.__version__}")
    print()

    # construct data module
    dm = construct_data_module(cfg)

    # create callbacks
    callbacks = construct_callbacks(cfg)

    # construct profiler
    profiler = construct_profiler(cfg)

    # create training/evaluator
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
    )

    # construct lighting module for train/test
    network = construct_network_module(cfg, dm.get_test_names())

    # train model
    if cfg.fit_model:
        trainer.fit(network, datamodule=dm)

    # test model
    if cfg.trainer.accelerator == "ddp":
        destroy_process_group()

        if not trainer.global_rank == 0:
            return

        trainer: pl.Trainer = instantiate(
            cfg.trainer,
            devices=min(1, cfg.trainer.get("devices", 0)),
            logger=logger,
            callbacks=callbacks,
            profiler=profiler,
        )

    if cfg.eval_model and cfg.fit_model:
        # this will select the checkpoint with the best validation metric
        # according to the ModelCheckpoint callback
        try:
            trainer.test(datamodule=dm)
        except:
            # there might not have been a validation epoch
            trainer.test(network, datamodule=dm)
    elif cfg.eval_model:
        # this will simply test the given model weights (when it's e.g.
        # manually loaded from a checkpoint)
        trainer.test(network, datamodule=dm)

    return None
