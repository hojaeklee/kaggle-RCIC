import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, mode = "train", site = 1, channels = [1, 2, 3, 4, 5, 6]):
        df = pd.read_csv(csv_file)
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

class LeNet(nn.Module):
    def __init__(self, num_classes = 1108, num_channels = 6):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = num_channels, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1)
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    train_acc = np.zeros(1)

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == 100:
            break
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += topk_accuracy(output, target)

        if batch_idx % args.log_interval == 0:
            print('Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item()))

    print("Train Epoch: {}; Train Loss: {:.4f}, Train Acc: {:.4f}%".format(epoch, train_loss/len(train_loader), train_acc[0]/len(train_loader)))

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

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
    parser = argparse.ArgumentParser(description='Argument Parser for LeNet5')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    path_data = "../data/"
    
    args = parse_arguments()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = ImageDataset(csv_file = os.path.join(path_data, 'train.csv'), img_dir = path_data)
    test_dataset = ImageDataset(csv_file = os.path.join(path_data, 'test.csv'), img_dir = path_data, mode = 'test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle = True, **kwargs)

    model = LeNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        # test(args, model, device, test_loader, criterion)

