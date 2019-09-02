import math
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import sys
import glob

import torchvision
from torchvision import transforms as T

class ImagesDS(D.Dataset):
    def __init__(self, csv_file, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6], is_cropped = False):
        
        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.is_cropped = is_cropped
        
        self.transforms = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(90), T.ToTensor()])
    
    def _load_img_as_tensor(self, file_name):
        with Image.open(file_name) as img:
            transformed_img = self.transforms(img)
            return transformed_img
    
    def _get_img_path(self, index, channel, cid=None):
        if cid is not None:
            experiment = self.records[index].experiment
            well = self.records[index].well
            plate = self.records[index].plate
            img_path = '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_cid{cid}_w{channel}*.png'])
            img_path = glob.glob(img_path)[0]
        else:
            experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
            img_path =  '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])
        return img_path
    
    
    def __getitem__(self, index):
        if self.is_cropped:
            s1n = self.records[index].site1_ncells 
            s2n = self.records[index].site2_ncells
            if self.site == 1 and s1n>0:
                sn = s1n
                cid = np.random.randint(0, s1n)
            elif self.site == 2 and s2n>0:
                sn = s2n
                cid = np.random.randint(0, s2n)
            else:
                #in case this example has NO cells just grab the next in line
                index += 1
                cid = 0
            if self.mode == 'train':
                paths = [self._get_img_path(index, ch, cid) for ch in self.channels]
                img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
                return img, int(self.records[index].sirna), int(self.records[index].group)
            else:
                images = torch.Tensor()
                for cid in range(sn):
                    paths = [self._get_img_path(index, ch, cid) for ch in 
                             self.channels]
                    img = torch.cat([self._load_img_as_tensor(img_path) for 
                                     img_path in paths])
                    images = torch.cat(images, img[None,:,:,:])
                return images, int(self.records[index].id_code), int(self.records[index].group)
        elif self.four_plates:
            s1n = self.records[index].site1_ncells 
            s2n = self.records[index].site2_ncells
            if self.site == 1 and s1n>0:
                sn = s1n
                cid = np.random.randint(0, s1n)
            elif self.site == 2 and s2n>0:
                sn = s2n
                cid = np.random.randint(0, s2n)
            else:
                #in case this example has NO cells just grab the next in line
                index += 1
                cid = 0
            if self.mode == 'train':
                paths = [self._get_img_path(index, ch, cid) for ch in self.channels]
                img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
                return img, int(self.records[index].group_target), int(self.records[index].group)
            else:
                images = torch.Tensor()
                for cid in range(sn):
                    paths = [self._get_img_path(index, ch, cid) for ch in 
                             self.channels]
                    img = torch.cat([self._load_img_as_tensor(img_path) for 
                                     img_path in paths])
                    images = torch.cat(images, img[None,:,:,:])
                return img, int(self.records[index].id_code), int(self.records[index].group)
                
        else:
            paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        if self.mode == 'train':
            return img, int(self.records[index].sirna), int(self.records[index].group)
        else:
            return img, self.records[index].id_code, int(self.records[index].group)

    def __len__(self):
        return self.len
