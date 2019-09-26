import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
import pandas as pd
from PIL import Image

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, mode = "train", site = 1, channels = [1, 2, 3, 4, 5, 6]):
        self.records = df.to_records(index = False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
    
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return transforms.ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        path = '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])
        return path

    def __getitem__(self, index):
        paths = [self._get_img_path(index, channel) for channel in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])

        if self.mode == "train":
            return img, self.records[index].sirna
        else:
            return img, self.records[index].id_code
    
    def __len__(self):
        return self.len

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    train_acc = np.zeros(1)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach().item()
        train_acc += topk_accuracy(output, target)

        if batch_idx % args.log_interval == 0:
            print('Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        del loss, output, target, data

    print("Train Epoch: {}; Train Loss: {:.4f}, Train Acc: {:.4f}%".format(epoch, train_loss/len(train_loader), train_acc[0]/len(train_loader)))

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def predict(model, device, loader):
    preds = np.empty(0)
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            
            output = model(x)
            idx = output.max(-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis = 0)
        return preds

def topk_accuracy(output, target, topk = (1,)):
    """Compute accuracy over top "k" predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size()[0]

        _, pred = output.topk(maxk, dim = 1, largest = True, sorted = True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

        return np.array(res)

def create_submission():
    submission = pd.read_csv(path_data + '/test.csv')
    submission['sirna'] = preds.astype(int)
    submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for ResNet18')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    path_data = "../data/raw"

    args = parse_arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    df = pd.read_csv(path_data + "/train.csv")
    df_train, df_val = train_test_split(df, test_size = 0.035, stratify = df.sirna, random_state = 42)
    df_test = pd.read_csv(path_data + "/test.csv")

    train_dataset = ImageDataset(df_train, img_dir = path_data, mode="train")
    val_dataset = ImageDataset(df_val, img_dir = path_data, mode="train")
    test_dataset = ImageDataset(df_test, img_dir = path_data, mode="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.test_batch_size, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle = False, **kwargs)

    # Defining model
    model = DenseNet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, val_loader, criterion)
