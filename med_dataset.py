import numpy as np
import nibabel as nib
from skimage.transform import resize as sk_resize

import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MedDataset(Dataset):

    CLASS_NAMES = ["Positive", "False Positive", "False Negative"]

    def __init__(self, split, model="VGG", data_dir="../data"):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        with open("seg/"+split+"_data.txt", "rb") as fp:
            self.file_list = pickle.load(fp)
        with open("seg/"+split+"_labels.txt", "rb") as fp:
            self.labels = pickle.load(fp)
        self.size = len(self.file_list)
        self.model_name = model

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fname = self.file_list[index]  # "xyz.nii"
        fpath = os.path.join(self.data_dir, fname)
        img_nib = nib.load(fpath)
        img = img_nib.get_fdata()   # converts to numpy array

        # add data augmentation here
        img = sk_resize(img, (128, 128, 32), order = 2)
        # img = img - img.mean()
        img = img / img.max()

        if self.model_name == "VGG" or self.model_name == "ResNet2D":
            img = sk_resize(img, (224, 224, 32), order = 1)
            image = torch.FloatTensor(np.transpose(img, (2, 0, 1)))
        if self.model_name == "ResNet3D" or self.model_name == "VGG3D":
            image3d = torch.FloatTensor(np.transpose(img, (2, 0, 1)))
            image = image3d.view(1, 32, 128, 128)

        label = int(self.labels[index])

        return image, label

def get_data_loader(train = True, batch_size=20, split='train', model = "VGG"):
    set = MedDataset(split, model)
    loader = DataLoader(set, batch_size = batch_size, shuffle = train, num_workers=6)
    return loader

def test(model, test_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss += criterion(output, target).item()
            pred = output.argmax(dim = 1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(test_loader)
    acc = correct / len(test_loader.dataset)

    return loss, acc
