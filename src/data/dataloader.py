import numpy as np
import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, train):
        if train:
            data = np.load("../../data/processed/train_images.npy", allow_pickle=True)
            labels = np.load("../../data/processed/train_labels.npy", allow_pickle=True)
        else:
            data = np.load("../../data/processed/test_images.npy", allow_pickle=True)
            labels = np.load("../../data/processed/test_labels.npy", allow_pickle=True)

        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)