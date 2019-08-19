import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from torchvision import models

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./ math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine

class ArcModule(nn.Module):
    """Code from https://github.com/ronghuaiyang/arcface-pytorch/blob/165e954f5ab1c6b34b99540690b981645b558f91/models/metrics.py"""
    def __init__(self, in_features, out_features, s = 64.0, m = 0.5):
        super(ArcModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, labels):
        cos_th = F.linear(input, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)
        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        onehot = torch.zeros(cos_th.size()).cuda()
        onehot.scatter_(1, labels, 1)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

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

        features = self.features(x)
        out = F.relu(features, inplace = True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim = 1)
        return out, features


class ArcResNet50(BaseModel):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained = True)
        trained_kernel = self.model.conv1.weight
        new_conv = nn.Conv2d(num_channels, 64, 7, 2, 3)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)] * 6, dim = 1)
        self.model.conv1 = new_conv
        in_features = self.model.fc.in_features
        self.model = torch.nn.Sequential(*(list(self.model.children())))[:-1]
        self.arc_margin_product = ArcMarginProduct(512, num_classes)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.dropout = nn.Dropout2d(0.25, inplace = True)
        feature_size = 1
        channel_size = 512
        self.fc1 = nn.Linear(in_features * feature_size * feature_size, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)
        
        # self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x, labels = None):
        features = self.model(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        out = self.arc_margin_product(features)
        
        return features

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
