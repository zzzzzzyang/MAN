# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
import pandas as pd
import random
from PIL import Image, ImageFile
import math
from util import *

ImageFile.LOAD_TRUNCATED_IAMGES = True

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform

        name_c = 0
        label_c = 1
        if phase == 'train':
            df_train = pd.read_csv(args.train_label_path, sep=' ', header=None)
            dataset = df_train
            print(dataset.groupby([1]).size())
        else:
            df_test = pd.read_csv(args.test_label_path, sep=' ', header=None)
            dataset = df_test
            print(dataset.groupby([1]).size())
            

        # notice the raf-db label starts from 1 while label of other dataset starts from 0
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [filp_image, add_g]
        self.file_paths = []

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)
            
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
        image = image[:, :, ::-1]
        
        if self.phase == 'train':
            if self.transform[0] is not None:
                image1 = self.transform[0](image)
                image2 = self.transform[1](image)
            return image1, image2, label, idx
        else:
            if self.transform is not None:
                image = self.transform(image)
            return image, label, idx
