#  Provides torch Datasets.
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from glob import glob
from PIL import Image


class ImageFolderDataset(Dataset):
    # Contructs a dataset from a folder with images
    def __init__(self, root, input_transform=None):
        self.image_filenames = [x for x in glob(root + '/*') if is_image_file(x.lower())]

        self.input_transform = input_transform

    def __getitem__(self, index):
        x = load_img(self.image_filenames[index])
        if self.input_transform:
            x = self.input_transform(x)
        return x, 0

    def __len__(self):
        return len(self.image_filenames)

class PKLDataset(Dataset):
    # Construct a dataset from a .pkl file
    def __init__(self, pkl_file):
        print('[*] Loading dataset from %s' % pkl_file)
        import pickle
        with open(pkl_file, 'rb') as fobj:
            self.images = pickle.load(fobj)
        print('[*] Dataset loaded')

    def __getitem__(self, index):
        x = self.images[index]
        return x, 0

    def __len__(self):
        return len(self.images)

class PTDataset(Dataset):
    # Construct a dataset from a .pt file
    def __init__(self, pt_file):
        print('[*] Loading dataset from %s' % pt_file)
        self.images = torch.load(pt_file)
        print('[*] Dataset loaded')

    def __getitem__(self, index):
        x = self.images[index]
        return x, 0

    def __len__(self):
        return len(self.images)


def is_image_file(filename):
    """Checks if a file is an image.
    
    Arguments:
        filename {str} -- File path.
    
    Returns:
        bool -- True if the path is PNG or JPG image.
    """    
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    """Loads an image from file.
    
    Arguments:
        filepath {str} -- Path of the image.
    
    Returns:
        PIL.Image.Image -- A PIL Image object.
    """    
    img = Image.open(filepath).convert('RGB')
    return img


def get_dataset(dset_name, data_root='./data', imsize=None, train=True):
    """Creates and returns a torch dataset.
    
    Arguments:
        dset_name {str} -- Name of the dataset.
    
    Keyword Arguments:
        data_root {str} -- Directory where datasets are stored (default: {'./data'}).
        imsize {int} -- Size of the image (default: {None}).
        train {bool} -- Whether to load the train split (default: {True}).
    
    Returns:
        Dataset -- A torch dataset,
    """    
    sizes = {'mnist': 32,
             'cifar10': 32,
             'stl10': 32,
             'celeba': 32,
             'celeba128': 128}
    assert dset_name in sizes.keys(), 'Unknown dataset {0}'.format(dset_name)
    if imsize is None:
        imsize = sizes[dset_name]
    # Resize, Center-crop, and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_with_antialias = transforms.Compose([
            transforms.Resize(imsize, Image.ANTIALIAS),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if dset_name == 'mnist':
        transform = transforms.Compose([
          transforms.Resize(imsize),
          transforms.CenterCrop(imsize),
          transforms.ToTensor(),
          transforms.Normalize([0.5], [0.5])])
        dataset = dset.MNIST(root=data_root, download=True,
                             train=train, transform=transform)
    elif dset_name == 'cifar10':
        dataset = dset.CIFAR10(root=data_root, download=True,
                               train=train, transform=transform)
    elif dset_name == 'stl10':
        dataset = dset.STL10(root=data_root, download=True,
                             split='unlabeled', transform=transform)
    elif dset_name == 'celeba':
        dataset = ImageFolderDataset(data_root, input_transform=transform_with_antialias)
        path = os.path.join(data_root, 'celeb.pkl')
        if os.path.exists(path):
            dataset = PKLDataset(path)
    elif dset_name == 'celeba128':
        dataset = ImageFolderDataset(data_root, input_transform=transform_with_antialias)
        path = os.path.join(data_root, 'celeba128.pt')
        if os.path.exists(path):
            dataset = PTDataset(path)
    return dataset
