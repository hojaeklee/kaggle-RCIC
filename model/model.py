import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from torchvision import models


class ArcMarginProduct(nn.Module):
    """Code from https://github.com/ronghuaiyang/arcface-pytorch/blob/165e954f5ab1c6b34b99540690b981645b558f91/models/metrics.py"""
    def __init__(self, in_features, out_features, s = 30.0, m = 0.50, easy_margin = False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device = input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class ResNeXT(BaseModel):
    def __init__(self):
        pass
    def forward(self, x):
        pass

class ArcDenseNet(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained = True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        self.arcface_classifier = ArcMarginProduct(in_features = 1024, out_features = num_classes, s = 30, m = 0.5)
        self.classifier = nn.Linear(1024, num_classes, bias = True)
        del preloaded

    def forward(self, x, label = None):
        features = self.features(x)
        out = F.relu(features, inplace = True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if label is not None:
            out = self.arcface_classifier(out, label)
        else:
            out = self.classifier(out)

        return out

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
        return out, features

class ResNet18(BaseModel):
    def __init__(self, num_classes = 1180, num_channels = 6):
        super().__init__()
        self.model_ft = torchvision.models.resnet18(pretrained=False)
        trained_kernel = self.model_ft.conv1.weight
        new_conv = nn.Conv2d(num_channels, 64, 7, 2, 3)
        with torch.no_grad():
                new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
        self.model_ft.conv1 = new_conv
        
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model_ft(x)
