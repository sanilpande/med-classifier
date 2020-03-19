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

    lr          = 0.0001
    train_size  = 32
    val_size    = 32
    test_size   = 32
    gamma       = 0.75
    epochs      = 25
    log         = 50

    train_loader = get_data_loader(train = True, batch_size = train_size,
                                split = 'train', model = models[model_id])
    val_loader = get_data_loader(train = False, batch_size = val_size,
                                split = 'val', model = models[model_id])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    # weight = torch.tensor([0.2, 0.3, 0.5]).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        count = 0
        current_loss = 0

        for data, target in train_loader:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            count += 1
            current_loss += loss.item()

            if count%log == 0:
                p = 100*count/len(train_loader)
                print("Epoch: {}    {:.2f}%    Loss: {}".format(epoch+1, p, loss.item()))

        scheduler.step()
        model.eval()

        val_loss, val_acc = test(model, val_loader, device)
        print("__________________________________________")
        print("\nValidation Acc:      ", val_acc)
        print("Validation Loss:     ", val_loss)
        print("\nTraining Loss:     ", current_loss/count)
        print("Learning Rate:       ", scheduler.get_lr()[0])
        print("__________________________________________")

        model.train()

        torch.save(model.state_dict(), models[model_id]+"_state_dict.pt")
        torch.save(model, models[model_id]+"_model.pt")

    # After training
    model.eval()
    test_loader = get_data_loader(train = False, batch_size = test_size,
                                split = 'test', model = models[model_id])

    _ , train_acc = test(model, train_loader, device)
    print("Final Train Accuracy: ", train_acc)

    _ , test_acc = test(model, test_loader, device)
    print("Final Accuracy: ", test_acc)
