import os
import torch
from base import BaseDataLoader
from dataset import ImagesDS

class RCICDataLoader(BaseDataLoader):
    """
    Loads ImageDS Dataset
    """
    def __init__(self, data_dir, batch_size, shuffle = True, drop_last=True, validation_split = 0.0, num_workers = 1, neg_ctrl = False, training = True, site = 1, is_cropped = False, four_plates = False):
        self.data_dir = data_dir
        self.is_cropped = is_cropped
        self.four_plates = four_plates
        if training:
            self.dataset1 = ImagesDS(os.path.join(self.data_dir, "train.csv"), 
                                     self.data_dir, site = 1, 
                                     is_cropped = self.is_cropped,
                                     four_plates = self.four_plates)
            self.dataset2 = ImagesDS(os.path.join(self.data_dir, "train.csv"), 
                                     self.data_dir, site = 2, 
                                     is_cropped = self.is_cropped,
                                     four_plates = self.four_plates)
            self.dataset = torch.utils.data.ConcatDataset([self.dataset1, 
                                                           self.dataset2])
        else:
            if neg_ctrl:
                self.dataset = ImagesDS(os.path.join(self.data_dir, 
                                                     "train_negative_controls.csv"), 
                                        self.data_dir, site = site)
            else:
                self.dataset = ImagesDS(os.path.join(self.data_dir, "test.csv"),
                                        self.data_dir, site = site, mode = 'test', 
                                        is_cropped = self.is_cropped,
                                        four_plates = self.four_plates)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
