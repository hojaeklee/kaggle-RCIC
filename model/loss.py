import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def bce_loss(output, target):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s = 30.0, m = 0.5):
        """
        """
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch = 0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device = 'cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma = 1
        loss = (loss1 + gamma * loss2)/(1 + gamma)
        return loss
