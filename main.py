
import io, json
import torch
from torch.autograd import Variable
import os
import argparse

import dle

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--config_file', required=True,
                    help='path to the json config file')
parser.add_argument('--output_dir', required=True,
                    help='output path of results')
parser.add_argument('--data_dir', required=True,
                    help='output path of results')
args = parser.parse_args()

with open(args.config_file) as json_file:
    c = json.load(json_file)

dataset = dle.dataset(c["dataset"])
transform = dle.transform(c["transforms"])

train_loader = torch.utils.data.DataLoader(
    dataset(args.data_dir, train=True, download=True,transform=transform),
    batch_size=c["batch_size"], shuffle=c["shuffle"])
test_loader = torch.utils.data.DataLoader(
    dataset(args.data_dir, train=False, transform=transform),
    batch_size=c["batch_size"], shuffle=c["shuffle"])


model = dle.neural_network(c["network"])()
criterion = dle.criterion(c["criterion"])
optim_func = dle.optim(c["optim"])
optimizer = optim_func(model.parameters(), lr=c["learning_rate"], momentum=c["momentum"])

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    return train_loss

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


losses = []

if not os.path.exists(args.output_dir):
  os.mkdir(args.output_dir);
if not os.path.exists(args.output_dir + '/models'):
  os.mkdir(args.output_dir + '/models');

for epoch in range(1, c["epochs"] + 1):
    train_loss = train(epoch)
    test_loss = test(epoch)
    losses.append({
      "train_loss": train_loss,
      "test_loss": test_loss
    })
    model_out_path = args.output_dir + '/models/model_' + str(epoch) + '.pth'
    torch.save(model, model_out_path)


with io.open(args.output_dir + '/losses.json', 'w', encoding='utf-8') as f:
  j_ = json.dumps(losses, ensure_ascii=False, indent=2, sort_keys=True)
  f.write(unicode(j_))

with io.open(args.output_dir + '/params.json', 'w', encoding='utf-8') as f:
  j_ = json.dumps(c, ensure_ascii=False, indent=2, sort_keys=True)
  f.write(unicode(j_))

