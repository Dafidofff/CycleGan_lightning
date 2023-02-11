import random 

from typing import List
from PIL import Image
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader



class MonetDataset(Dataset):
    """ Monet dataset for training a cycle GAN model generate 
    monet style images from photos. 

    Source: https://www.kaggle.com/code/bootiu/cyclegan-pytorch-lightning/data

    Args:
        base_img_paths (list): list of base image paths
        style_img_paths (list): list of style image paths
        transform (callable): transform to apply to the image
        split (str): train or val

    Returns:
        base_img (tensor): base image
        style_img (tensor): style image
    """

    def __init__(self, base_img_paths: List[Path], style_dir_path: List[Path],  transform, split='train'):

        # List of Path objects
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_dir_path

        # Save additional stuff
        self.transform = transform
        self.split = split

    def __len__(self):
        return min([len(self.base_img_paths), len(self.style_img_paths)])

    def __getitem__(self, idx):        
        base_img_path = self.base_img_paths[idx]
        style_img_path = self.style_img_paths[idx]

        base_img = Image.open(base_img_path)
        style_img = Image.open(style_img_path)

        if self.transform:
            base_img = self.transform(base_img, self.split)
            style_img = self.transform(style_img, self.split)

        return base_img, style_img


class MonetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, split='train', seed=0):
        super(MonetDataModule, self).__init__()
        self.data_dir = data_dir
        self.base_dir_path = data_dir / 'photo_jpg'
        self.style_dir_path = data_dir / 'monet_jpg'

        self.transform = transform
        self.batch_size = batch_size
        self.split = split
        self.seed = seed

    def prepare_data(self):
        self.base_img_paths = list(self.style_dir_path.glob('*.jpg'))
        self.style_img_paths = list(self.style_dir_path.glob('*.jpg'))


    def train_dataloader(self):
        # random.seed()
        # random.shuffle(self.base_img_paths)
        # random.shuffle(self.style_img_paths)
        # random.seed(self.seed)
        self.train_dataset = MonetDataset(self.base_img_paths, self.style_img_paths, self.transform, self.split)
        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True
                         )