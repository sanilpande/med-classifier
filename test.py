import numpy as np
import torch
import torchvision

import med_dataset
from med_dataset import get_data_loader, test

from res3d import resnet34, resnet50
from models import VGG3D, VGG, ResNet2D


if __name__ == '__main__':

    models = [ "VGG3D", "ResNet3D", "ResNet2D", "VGG"]
    model_id = 0    # Change this to correspond to the model in the list
    if model_id == 0:
        model = VGG3D()
    elif model_id == 1:
        model = resnet34()
    elif model_id == 2:
        model = ResNet2D(3)
    elif model_id == 3:
        model = VGG(3)

    train_size  = 15
    test_size   = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)

    model = torch.load("model_m.pt")
    model.eval()

    train_loader = get_data_loader(train = True, batch_size = train_size,
                                split = 'train', model = models[model_id])
    test_loader = get_data_loader(train = False, batch_size = test_size,
                                split = 'test', model = models[model_id])

    _ , train_acc = test(model, train_loader, device)
    print("Final Train Accuracy: ", train_acc)

    _ , test_acc = test(model, test_loader, device)
    print("Final Accuracy: ", test_acc)
