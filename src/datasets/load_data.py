from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision import io 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from src.datasets import preprocess



class DatasetFromCsv(Dataset):
    '''
    Create dataset from annotations_file.csv: path:str, label:number

    Usage:
        train_ds = load_data.DatasetFromFolder("./labels_train.csv", transform, target_transform)
    '''
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        super(DatasetFromCsv,self).__init__()
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0] 
        
        image = io.read_image(img_path, mode=io.image.ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class DatasetFromInternet:
    '''
    Dataset from TorchVision
    https://pytorch.org/vision/stable/datasets.html

    Usage:
            train_set = load_data.DatasetFromInternet('mnist', stored_dir='./ds',train=True download=False)
    '''
    def __init__(self, name, stored_dir, train, download=False):
        name2ds = {'mnist': datasets.MNIST(stored_dir, train=train, download=download),
                   'fashion-mnist': datasets.FashionMNIST(stored_dir, train=train, download=download),
                   'cifar10': datasets.CIFAR10(stored_dir, train=train, download=download),
                    'cifar100': datasets.CIFAR100(stored_dir, train=train, download=download),
                    'imagenet': datasets.ImageNet(stored_dir, train=train, download=download),
                    'coco': datasets.CocoDetection(stored_dir, train=train, download=download),
                    'voc': datasets.VOCDetection(stored_dir, train=train, download=download)}
        self = name2ds[name]

class DatasetFromFolder(datasets.ImageFolder):
    '''
        Dataset from folder. 
    ```
    ├── train
    │   ├── class1
    |      ├── 1.jpg
    │      ├── 2.jpg
    │   ├── class2
    |      ├── 1.jpg
    │      ├── 2.jpg
    ├── val
    │   ├── class1
    |      ├── 1.jpg
    │      ├── 2.jpg
    │   ├── class2
    |      ├── 1.jpg
    │      ├── 2.jpg
    ├── test
    │   ├── class1
    |      ├── 1.jpg
    │      ├── 2.jpg
    │   ├── class2
    |      ├── 1.jpg
    │      ├── 2.jpg
    ```
        Usage:
            train_ds = load_data.DatasetFromFolder("./resources/data/train", transform, target_transform)
    '''
    def __init__(self, folder_dir, transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__(folder_dir, transform=transform, target_transform=target_transform)


def create_dataloader(dataset, batch_size, workers=4, shuffle=True):
    '''
    Usage:
        train_loader = load_data.create_dataloader(train_ds, batch_size=32, shuffle=True)
    '''
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle)




