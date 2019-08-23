import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from torchvision import models

class DenseNet(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_classes, bias = True)
        del preloaded
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace = True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim = 1)
        return out

class ResNet50(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained = True)
        trained_kernel = self.model.conv1.weight
        new_conv = nn.Conv2d(num_channels, 64, 7, 2, 3)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
        self.model.conv1 = new_conv
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        out = self.model(x)
        out = F.log_softmax(out, dim = 1)
        return out

class DenseNet201(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        preloaded = torchvision.models.densenet201(pretrained = True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1920, num_classes, bias = True)
        del preloaded

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace = True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim = 1)
        return out

class ResNet152(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        preloaded = torchvision.models.resnet152(pretrained = True)
        trained_kernel = preloaded.conv1.weight
        new_conv = nn.Conv2d(num_channels, 64, 7, 2, 3)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
        preloaded.conv1 = new_conv
        num_ftrs = preloaded.fc.in_features
        preloaded.fc = nn.Linear(num_ftrs, num_classes)

        self.model = preloaded

    def forward(self, x):
        out = self.model(x)
        out = F.log_softmax(out, dim = 1)
        return out

class ResNext101_32x8d(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        preloaded = torchvision.models.resnet152(pretrained = True)
        trained_kernel = preloaded.conv1.weight
        new_conv = nn.Conv2d(num_channels, 64, 7, 2, 3)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
        preloaded.conv1 = new_conv
        num_ftrs = preloaded.fc.in_features
        preloaded.fc = nn.Linear(num_ftrs, num_classes)

        self.model = preloaded

    def forward(self, x):
        out = self.model(x)
        out = F.log_softmax(out, dim = 1)
        return out

