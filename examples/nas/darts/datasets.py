# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import os
import natsort
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, random_split

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

class CustomDataSet(Dataset):
    def __init__(self, csv_file,main_dir, transform):
        self.label_frame = pd.read_csv(csv_file,usecols=["Label"])
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        labels = self.label_frame.iloc[idx, 0]
        return (tensor_image,labels)

def get_dataset(cls, cutout_length=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose(transf + normalize + cutout)
    valid_transform = transforms.Compose(normalize)
    
    #section for AOI
    img_folder_path="/home/satyajit/nni/AOI/aoi/train_images"
    label_file="/home/satyajit/nni/AOI/aoi/train.csv"

    trsfm = transforms.Compose([transforms.Resize((32,32)),transforms.CenterCrop(32),transforms.ToTensor()])

    my_dataset = CustomDataSet(label_file,img_folder_path,transform=trsfm)

    trainset, valset = random_split(my_dataset, [2026, 502])

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    elif cls=="aoi":
        dataset_train =trainset 
        dataset_valid =valset
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid
