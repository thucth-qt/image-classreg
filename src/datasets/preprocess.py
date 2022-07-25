from torchvision import transforms

import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import *

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

input_size = (224,224)
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)

    ##########################################################################
    #                       TorchVision preprocessing                        #
    ##########################################################################
    # "Compose",
    # "ToTensor",
    # "PILToTensor",
    # "ConvertImageDtype",
    # "ToPILImage",
    # "Normalize",
    # "Resize",
    # "CenterCrop",
    # "Pad",
    # "Lambda",
    # "RandomApply",
    # "RandomChoice",
    # "RandomOrder",
    # "RandomCrop",
    # "RandomHorizontalFlip",
    # "RandomVerticalFlip",
    # "RandomResizedCrop",
    # "FiveCrop",
    # "TenCrop",
    # "LinearTransformation",
    # "ColorJitter",
    # "RandomRotation",
    # "RandomAffine",
    # "Grayscale",
    # "RandomGrayscale",
    # "RandomPerspective",
    # "RandomErasing",
    # "GaussianBlur",
    # "InterpolationMode",
    # "RandomInvert",
    # "RandomPosterize",
    # "RandomSolarize",
    # "RandomAdjustSharpness",
    # "RandomAutocontrast",
    # "RandomEqualize",

#Label
target_transform = transforms.Lambda(lambda x: int(x))


#Train samples
transform_train = transforms.Compose([
    transforms.RandomCrop(),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    # transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
    ])


#Val samples
transform_val = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
    ])
