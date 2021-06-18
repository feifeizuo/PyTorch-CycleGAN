import glob
import random
import os
import numpy as np
import torch
import torchvision.transforms.functional
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ToothWhiteningDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files_A = glob.glob(os.path.join(root, '%s/*/A1.JPG' % mode))
        self.files_B = glob.glob(os.path.join(root, '%s/*/B1.JPG' % mode))
        print('build tooth whitening dataset done.')

    def __getitem__(self, index):

        img = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        target = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(256, 256))

        img = torchvision.transforms.functional.crop(img,i,j,h,w)
        target = torchvision.transforms.functional.crop(target,i,j,h,w)

        # Random flipping

        if random.random() > 0.5:
            img = torchvision.transforms.functional.hflip(img)
            target = torchvision.transforms.functional.hflip(target)

        if random.random() > 0.5:
            img = torchvision.transforms.functional.vflip(img)
            target = torchvision.transforms.functional.vflip(target)

        path = self.files_A[index % len(self.files_A)]
        return {'A': img, 'B': target, 'Path': os.path.dirname(path)}


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

