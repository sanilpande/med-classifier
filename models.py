import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def ResNet2D(num_classes=3):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.conv1 = nn.Conv2d(42, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(512, num_classes)
    return resnet

def VGG(num_classes=3):
    model = torchvision.models.vgg16_bn(pretrained=True)
    model.features[0] = nn.Conv2d(42, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    return model

class VGG3D(nn.Module):
    def __init__(self, num_classes=3, inp_size=128, c_dim=1):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv3d(in_channels=c_dim, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')   # padding zero since valid
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode='zeros')

        self.flat_dim =  8192
        self.pool = nn.MaxPool3d(2,2)

        self.fc1 = nn.Linear(self.flat_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3)
        self.fc4 = nn.Linear(10, 3)

        self.dropout = nn.Dropout(p=0.5)
        self.nonlinear = nn.ReLU()

        self.bn_conv1 = nn.BatchNorm3d(32)
        self.bn_conv2 = nn.BatchNorm3d(64)
        self.bn_conv3 = nn.BatchNorm3d(128)
        self.bn_conv4 = nn.BatchNorm3d(256)
        self.bn_conv5 = nn.BatchNorm3d(512)

    def forward(self, x):

        N = x.size(0)

        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn_conv4(x)
        x = self.nonlinear(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.bn_conv5(x)
        x = self.nonlinear(x)
        x = self.pool(x)

        flat_x = x.view(N, self.flat_dim)

        out = self.fc1(flat_x)
        out = self.dropout(out)
        out = self.nonlinear(out)
        out = self.fc2(out)
        out = self.nonlinear(out)
        out = self.dropout(out)
        out = self.fc3(out)

        return out
