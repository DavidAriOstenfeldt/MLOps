import click
import numpy as np
import torch
from model import MyAwesomeModel
from torch.utils.data import DataLoader, Dataset


class dataset(Dataset):
    def __init__(self, train):
        if train:
            data = np.load("data/processed/train_images.npy", allow_pickle=True)
            labels = np.load("data/processed/train_labels.npy", allow_pickle=True)
        else:
            data = np.load("data/processed/test_images.npy", allow_pickle=True)
            labels = np.load("data/processed/test_labels.npy", allow_pickle=True)

        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.argument("model_checkpoint")
@click.argument("test_path")
def evaluate(model_checkpoint, test_path):
    print("eEvaluating until ceiling hit")
    print(model_checkpoint)

    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)

    test_set = dataset(train=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    with torch.no_grad():
        model.eval()

        accuracy = 0
        for images, labels in test_loader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy_ = torch.mean(equals.type(torch.FloatTensor))
            accuracy += accuracy_.item()

        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate()
