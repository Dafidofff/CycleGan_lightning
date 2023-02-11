from PIL import Image
from torch.utils.data import Dataset


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

    def __init__(self, base_img_paths, style_img_paths,  transform, split='train'):
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
        self.transform = transform
        self.split = split

    def __len__(self):
        return min([len(self.base_img_paths), len(self.style_img_paths)])

    def __getitem__(self, idx):        
        base_img_path = self.base_img_paths[idx]
        style_img_path = self.style_img_paths[idx]
        base_img = Image.open(base_img_path)
        style_img = Image.open(style_img_path)

        base_img = self.transform(base_img, self.split)
        style_img = self.transform(style_img, self.split)

        return base_img, style_img