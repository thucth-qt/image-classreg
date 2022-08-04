from torchvision import transforms

import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import *

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from src.constants import constants
    ##########################################################################
    #                       TorchVision preprocessing                        #
    ##########################################################################
    # "Compose",
    # "ToTensor",
    # "PILToTensor",
    # "ConvertImageDtype",
    # "ToPILImage", : this is because torchvision just support for PILImage
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
def create_target_tf():
    # transform_target = transforms.Lambda(lambda x: int(x))
    transform_target = transforms.Lambda(lambda x: torch.tensor([x], dtype=torch.float32))
    return transform_target

#Train samples
def create_train_tf(input_size=None, means=None, stds=None):
    if not input_size:
        input_size=constants.INPUT_SIZE
    if not means:
        means=constants.IMAGENET_DEFAULT_MEAN
    if not stds:
        stds=constants.IMAGENET_DEFAULT_STD

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        # transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        ])
    return transform_train

#Val samples
def create_val_tf(input_size=None, means=None, stds=None):
    if not input_size:
        input_size=constants.INPUT_SIZE
    if not means:
        means=constants.IMAGENET_DEFAULT_MEAN
    if not stds:
        stds=constants.IMAGENET_DEFAULT_STD
        
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        ])
    return transform_val


def invert_val_tf(input, means=None, stds=None):
    if not means:
        means=constants.IMAGENET_DEFAULT_MEAN
    if not stds:
        stds=constants.IMAGENET_DEFAULT_STD
    
    return transform_val(input)
