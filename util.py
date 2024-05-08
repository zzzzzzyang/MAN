import cv2
import random
import torch
import numpy as np
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from typing import Tuple

#add gaussian noise
def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

#flip image
def filp_image(image_array):
    return cv2.flip(image_array, 1)

#set random seed to ensure the results can be reproduced
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


