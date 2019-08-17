import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from torchvision import models

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Arcface(nn.Module):
    """Code from https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py"""
    def __init__(self, embedding_size=1024, classnum=1108, s = 64., m = 0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))

        # Initial kernel
        self.kernel.data.uniform_(-1, 1).renorm(2, 1, 1e-5).mul(1e5)
        self.m = m 
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m 
        self.threshold = math.cos(math.pi - m)

    def forward(self, embeddings, label):
        # weights norm
        nB = len(embeddings)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        # cos(theta + m)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) 
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s
        return output

class DenseNet121_Extractor(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        del preloaded
        """
        if isinstance(self.features.last_linear, nn.Conv2d):
            in_features = self.features.last_linear.in_channels
        else:
            in_features = self.features.last_linear.in_features
        """
        in_features = 1024
        feature_size = 16
        channel_size = 512
        # self.bn1 = nn.BatchNorm2d(in_features)
        # self.dropout = nn.Dropout2d(0.5, inplace = True)
        self.fc1 = nn.Linear(in_features * feature_size * feature_size, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)

    def forward(self, x):
        features = self.features(x)
        # features = self.bn1(features)
        # features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)

        return features

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

class DenseNet201(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        preloaded = torchvision.models.densenet201(pretrained = True)
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
