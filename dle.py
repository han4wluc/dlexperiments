
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
# def SGD:
#   return optim.SGD

def transform(transforms_json):
  tlist = []
  for transform_json in transforms_json:
    if transform_json["name"] == 'ToTensor':
      tlist.append(transforms.ToTensor())
    if transform_json["name"] == 'Normalize':
      mean = transform_json["mean"]
      std = transform_json["std"]
      tlist.append(transforms.Normalize((mean,), (std,)))
  return transforms.Compose(tlist)

def optim(name):
  if name == 'SGD':
    return torch.optim.SGD
  print('error optim')

def criterion(name):
  if name == 'nll_loss':
    return F.nll_loss
  print('error criterion')

def dataset(name):
  if name == 'MNIST':
    return datasets.MNIST
  print('error dataset')


class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def neural_network(name):
  if name == 'CNN_1':
    return CNN_1
  print('error nn')
  


