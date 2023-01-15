import argparse
import logging
import os.path
import sys
from typing import Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model import MyAwesomeModel
from torch import optim
from torch.utils.data import DataLoader, Dataset
import hydra
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

log = logging.getLogger(__name__)

class dataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = images
        self.labels = labels

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[item].float(), self.labels[item]

    def __len__(self) -> int:
        return len(self.data)


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def train(cfg) -> None:
    log.info("Training day and night")
    model_hparams = cfg.model
    train_hparams = cfg.training

    torch.manual_seed(train_hparams.hyperparameters.seed)

    model = MyAwesomeModel(model_name=model_hparams.hyperparameters.model_name,
                           lr=train_hparams.hyperparameters.lr,)


    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="val_loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor='train_loss',
        patience=train_hparams.hyperparameters.patience,
        verbose=True,
        mode="min"
    )

    accelerator = "gpu" if train_hparams.hyperparameters.cuda else "cpu"
    wandb_logger = WandbLogger(
        project="mlops_mnist", entity="personal", log_model="all"
    )

    for key, val in train_hparams.hyperparameters.items():
        wandb_logger.experiment.config[key] = val

    trainer = Trainer(
        devices=1,
        accelerator=accelerator,
        max_epochs=train_hparams.hyperparameters.epochs,
        limit_train_batches=train_hparams.hyperparameters.limit_train_batches,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger
    )

    log.info(f"device (accelerator): {accelerator}")

    with open(train_hparams.hyperparameters.train_data_path, "rb") as handle:
        train_image_data = np.load(handle, allow_pickle=True)

    with open(train_hparams.hyperparameters.train_labels_path, "rb") as handle:
        train_labels = np.load(handle, allow_pickle=True)

    train_data = dataset(train_image_data, train_labels.long())
    train_loader = DataLoader(
        train_data,
        batch_size=train_hparams.hyperparameters.batch_size,
        num_workers=1,
        shuffle=True
    )

    with open(train_hparams.hyperparameters.val_data_path, "rb") as handle:
        val_image_data = np.load(handle, allow_pickle=True)

    with open(train_hparams.hyperparameters.val_labels_path, "rb") as handle:
        val_labels = np.load(handle, allow_pickle=True)

    val_data = dataset(val_image_data, val_labels.long())
    val_loader = DataLoader(
        val_data,
        batch_size=train_hparams.hyperparameters.batch_size,
        num_workers=1
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



if __name__ == "__main__":
    train()
