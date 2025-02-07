import os
import torch
from base import BaseDataLoader
from dataset import ImagesDS

class RCICDataLoader(BaseDataLoader):
    """
    Loads ImageDS Dataset
    """
    def __init__(self, data_dir, batch_size, shuffle = True, validation_split = 0.0, num_workers = 1, neg_ctrl = False, training = True, site = 1):
        self.data_dir = data_dir

        if training:
            self.dataset1 = ImagesDS(os.path.join(self.data_dir, "train.csv"), self.data_dir, site = 1)
            self.dataset2 = ImagesDS(os.path.join(self.data_dir, "train.csv"), self.data_dir, site = 2)
            self.dataset = torch.utils.data.ConcatDataset([self.dataset1, self.dataset2])
        else:
            if neg_ctrl:
                self.dataset = ImagesDS(os.path.join(self.data_dir, "train_negative_controls.csv"), self.data_dir, site = site)
            else:
                self.dataset = ImagesDS(os.path.join(self.data_dir, "test.csv"), self.data_dir, site = site, mode = 'test')

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

