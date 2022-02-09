import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

import random



def load_dataset(data_root, y_index=2, class_ratio=0.5, dataset='mnist', batch_size = 128):
    if dataset == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)), ]
        )
        train_dataset = datasets.MNIST(root=data_root, download=True, train=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_root, download=True, train=False, transform=transform)

    train_dataset = Imabalance(train_dataset, y_index=y_index, class_ratio=class_ratio)

    train_loader = dset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = dset.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_per_class = {y: 0 for y in range(10)}
    for x,y in train_dataset:
        num_per_class[y] += 1

    return train_loader, test_loader, num_per_class


class Imabalance(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_dataset, y_index, class_ratio):
        """
        Args:
            train_dataset (Dataset): torchvision dataset object
            y_index (Integer): Class index with imbalance
            class_ratio (Float): ratio of the minority class wrt to most frequent class
        """
        self.num_per_class = {y: 0 for y in range(10)}

        self.train_dataset = train_dataset

        self.subdataset = []
        self.imb_dataset = []

        for x, y in self.train_dataset:
            if y != y_index:
                self.subdataset.append((x, y))
                self.num_per_class[y] += 1
            else:
                self.imb_dataset.append((x, y))

        num_samples_y = int(max(self.num_per_class.values()) * class_ratio)
        if class_ratio != 1:
            self.imb_dataset = random.sample(self.imb_dataset, num_samples_y)

        self.train_dataset = self.subdataset + self.imb_dataset
        random.shuffle(self.train_dataset)

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        im, label = self.train_dataset[idx]

        return im, label










