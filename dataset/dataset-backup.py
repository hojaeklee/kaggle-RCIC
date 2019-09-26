import math
from PIL import Image

import pandas as pd
import numpy as np
import torch
import torch.utils.data as D

import torchvision
from torchvision import transforms as T

class ImagesDS(D.Dataset):
    def __init__(self, csv_file, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):
        
        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        
        self.transforms = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(90)])
        # self.transforms = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(90), T.ToTensor()])
        # self.transforms = T.Compose([T.RandomResizedCrop(512), T.RandomHorizontalFlip(), T.ColorJitter(brightness = [0.6, 1.4], contrast = [0.6, 1.4], saturation = [0.6, 1.4], hue = 0.5), T.RandomRotation(90), T.ToTensor()])
        # self.normalize = T.Normalize(mean = [0.485, 0.485, 0.456, 0.456, 0.406, 0.46], std = [0.229, 0.229, 0.225, 0.225, 0.224, 0.224])

    def _load_img_as_tensor(self, file_name):
        
        img = np.asarray(Image.open(file_name))
        img = (img - img.mean()) / (img.std())
        img = Image.fromarray(img)
        transformed_img = self.transforms(img)
        transformed_img = torch.from_numpy(np.asarray(transformed_img))
        transformed_img = transformed_img.unsqueeze(0)
        """
        with Image.open(file_name) as img:
            transformed_img = self.transforms(img)
            return transformed_img
        """
        return transformed_img

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        # print(img.shape)

        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len
