import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)

def bce_loss(output, target):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

