# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
import torchvision.transforms as transforms
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    transform_norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0, 1)]
    )

    # Prepare training data
    train = []
    for i in range(5):
        train.append(np.load(f"{input_filepath}/train_{i}.npz", allow_pickle=True))

    data = np.concatenate([img["images"] for img in train]).reshape(-1, 1, 28, 28)
    targets = torch.tensor(np.concatenate([img["labels"] for img in train]))

    data = np.squeeze(data)
    data = transform_norm(data)
    data = data.reshape(-1, 1, 28, 28)

    # Prepare test data
    test = np.load(f"{input_filepath}/test.npz", allow_pickle=True)
    test_data = test["images"].reshape(-1, 1, 28, 28)
    test_targets = torch.tensor(test["labels"])
    test_data = np.squeeze(test_data)
    test_data = transform_norm(test_data)
    test_data = test_data.reshape(-1, 1, 28, 28)

    np.save(f"{output_filepath}/train_images.npy", data, allow_pickle=True)
    np.save(f"{output_filepath}/train_labels.npy", targets, allow_pickle=True)
    np.save(f"{output_filepath}/test_images.npy", test_data, allow_pickle=True)
    np.save(f"{output_filepath}/test_labels.npy", test_targets, allow_pickle=True)

    logger.info("finished processing raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
