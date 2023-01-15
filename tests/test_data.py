import os
import os.path
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from tests import _PATH_DATA

# dataset = MNIST(...)
# assert len(dataset) == N_train for training and N_test for test
# assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
# assert that all labels are represented


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


def data_load(filepath):
    with open(filepath + "_images.npy", "rb") as fb:
        images = np.load(fb)
    with open(filepath + "_labels.npy", "rb") as fb:
        labels = np.load(fb)
    return dataset(images, labels)


class TestClass:
    N_train = 25000
    N_test = 5000

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PATH_DATA, "processed/train_images.npy")),
        reason="Data files not found",
    )
    def test_train_data(self):
        # load data
        dataset = data_load(os.path.join(_PATH_DATA, "processed/train"))
        # Ensure correct data-size
        assert len(dataset) == self.N_train, "Data is incomplete"
        # Ensure correct data shape
        assert dataset.data.shape == (
            self.N_train,
            1,
            28,
            28,
        ), "Data is of wrong shape"
        # Ensure number of labels correspond to number of images
        assert len(dataset.labels) == len(
            dataset.data
        ), "Number of labels does not correspond to number of data-points"
        # Ensure all labels are represented
        assert torch.tensor(np.sum(np.unique(dataset.labels))) == torch.tensor(
            45
        ), "Labels do not represent all classes."

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PATH_DATA, "processed/test_images.npy")),
        reason="Data files not found",
    )
    def test_test_data(self):
        # load data
        dataset = data_load(os.path.join(_PATH_DATA, "processed/test"))
        # Ensure correct data-size
        assert len(dataset) == self.N_test, "Data is incomplete"
        # Ensure correct data shape
        assert dataset.data.shape == (
            self.N_test,
            1,
            28,
            28,
        ), "Data is of wrong shape"
        # Ensure number of labels correspond to number of images
        assert len(dataset.labels) == len(
            dataset.data
        ), "Number of labels does not correspond to number of data-points"
        # Ensure all labels are represented
        assert torch.tensor(np.sum(np.unique(dataset.labels))) == torch.tensor(
            45
        ), "Labels do not represent all classes."
