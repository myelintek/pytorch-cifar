'''Train CIFAR10 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from models import *
from utils import progress_bar
from mlsteam import stparams

network_map = {
    'ResNet18': ResNet18,
    'PreActResNet18': PreActResNet18,
    'GoogLeNet': GoogLeNet,
    'DenseNet121': DenseNet121,
    'ResNeXt29_2x64d': ResNeXt29_2x64d,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DPN92': DPN92,
    'ShuffleNetG2': ShuffleNetG2,
    'SENet18': SENet18,
    'ShuffleNetV2': ShuffleNetV2,
    'EfficientNetB0': EfficientNetB0,
    'RegNetX_200MF': RegNetX_200MF
}

parser = argparse.ArgumentParser(description='Train cifar10 dataset in Pytorch')
parser.add_argument('--num_epochs', type=int, default=200, required=False, help='train epochs')
parser.add_argument('--download', type=int, default=0, required=False, help='download dataset')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
learning_rate = stparams.get_value("learning_rate", 0.1)
momentum = stparams.get_value("momentum", 0.9)
weight_decay = stparams.get_value("weight_decay", 5e-4)
num_epochs = stparams.get_value("num_epochs", args.num_epochs)
network = stparams.get_value("network", 'RegNetX_200MF')
test_bs = stparams.get_value("test_batch_size", 100)
test_worker = stparams.get_value("test_worker", 2)
dl_dataset = True if args.download else False


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(
    root='/mlsteam/data/cifar10', train=True, download=dl_dataset, transform=transform_train)

# Hyperparameter example
train_bs = stparams.get_value("train_batch_size", 128)
train_worker = stparams.get_value("train_worker", 2)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=train_bs, shuffle=True, num_workers=train_worker)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/mlsteam/data/cifar10', train=False, download=dl_dataset, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_bs, shuffle=False, num_workers=test_worker)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

default_cp = "{}_lr{}_bs{}_epochs{}.pth".format(network, learning_rate, train_bs, num_epochs)
checkpoint_path = stparams.get_value("checkpoint_path", default_cp)
if isinstance(checkpoint_path, list):
    checkpoint_path = default_cp

# Model
print('==> Building model..')
net = network_map.get(network)()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=momentum, weight_decay=weight_decay)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}'.format(checkpoint_path))
        best_acc = acc
        
def each_acc():
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(8):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("=============\nEach accuracy:")
    for i in range(10):
        print('Accuracy of %5s : %2.1f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def info():
    print('====================')
    print('| network:{}'.format(network))
    print('| num_epochs:{}'.format(num_epochs))
    print('| learning_rate:{}'.format(learning_rate))
    print('| momentum:{}'.format(momentum))
    print('| weight_decay:{}'.format(weight_decay))
    print('| dataset:{}'.format('cifar10'))
    print('| batch_size-> train:{}, test:{}'.format(train_bs, test_bs))
    print('| dataset_worker-> train:{}, test:{}'.format(train_worker, test_worker))
    print('| checkpoint path: {}'.format(checkpoint_path))

info()
for epoch in range(int(num_epochs)):
    train(epoch)
    test(epoch)
    
each_acc()

