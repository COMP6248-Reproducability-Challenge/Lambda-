import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageNette(Dataset):
    
    def __init__(self, csv_file, root_dir, noisy_level=0, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_path_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.noisy_level = noisy_level
        self.train = train
        self.train_map = {
            'n02979186': 0,
            'n03417042': 1,
            'n01440764': 2,
            'n02102040': 3,
            'n03028079': 4,
            'n03888257': 5,
            'n03394916': 6,
            'n03000684': 7,
            'n03445777': 8,
            'n03425413': 9
        }
        self.val_map ={
            'n02979186': 0,
            'n03417042': 1,
            'n01440764': 2,
            'n02102040': 3,
            'n03028079': 4,
            'n03888257': 5,
            'n03394916': 6,
            'n03000684': 7,
            'n03445777': 8,
            'n03425413': 9
        }


    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.image_path_list['path'][idx])
        image = Image.open(img_path).convert('RGB')
        labels_head = 'noisy_labels_'+str(self.noisy_level)
        if self.train:
            label = self.train_map[self.image_path_list[labels_head][idx]]
        else:
            label = self.val_map[self.image_path_list[labels_head][idx]]
        label = torch.from_numpy(np.array([label]))

        if self.transform:
            image = self.transform(image)

        sample =  [image, label]
        return sample
