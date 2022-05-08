import pandas as pd
import torch as torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from PIL import Image


class Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file,shape):
        self.dataset_frame = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.dataset_frame)

    def __getitem__(self, index):
        'Generates one sample of data'
        file = self.dataset_frame.iloc[index,0]
        X = Image.open(file)
        X = self.transform(X)
        y = self.dataset_frame.iloc[index,1]

        return X, y