import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    train = []
    for i in range(5):
        train_i = np.load(f"data/corruptmnist/train_{i}.npz", allow_pickle=True)
        train.append(train_i)

    # Rewrite ...
    train_data = train.get("images")
    train_labels = train.get("labels")

    test = torch.randn(10000, 784) 
    return train, test

train, test = mnist()

print("duh")