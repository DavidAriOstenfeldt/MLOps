"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out
        # Alternative: Use torchvision.datasets.ImageFolder, see line 46
        self.transform = transform
        
    def __len__(self):
        return None # TODO: fill out
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        return self.transform(img)



def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='data/lfw', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = ImageFolder(args.path_to_folder, lfw_trans) # LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        plt.rcParams["savefig.bbox"] = 'tight'
        for (batch, labels) in dataloader:
            grid = make_grid(batch)
            show(grid)
            plt.show()
            # plt.savefig("figures/visualization.png")
            break

        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)

        # x = np.arange(args.num_workers)
        x = np.arange(len(res))
        mu = np.mean(res)
        std = np.std(res)
        plt.errorbar(x, res, yerr=std)
        plt.show()

        print(f'Timing: {np.mean(res)}+-{np.std(res)}')
