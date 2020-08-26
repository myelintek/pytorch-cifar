'''Eval CIFAR10 with PyTorch.'''
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = stparams.get_value("learning_rate", 0.1)
momentum = stparams.get_value("momentum", 0.9)
weight_decay = stparams.get_value("weight_decay", 5e-4)
num_epochs = stparams.get_value("num_epochs", 200)
network = stparams.get_value("network", 'RegNetX_200MF')
train_bs = stparams.get_value("train_batch_size", 128)
train_worker = stparams.get_value("train_worker", 2)
test_bs = stparams.get_value("test_batch_size", 100)
test_worker = stparams.get_value("test_worker", 2)
default_cp = "{}_lr{}_bs{}_epochs{}.pth".format(network, learning_rate, train_bs, num_epochs)
checkpoint_path = stparams.get_value("checkpoint_path", default_cp)
if isinstance(checkpoint_path, list):
    checkpoint_path = default_cp

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='/mlsteam/input/cifar10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_bs, shuffle=False, num_workers=test_worker)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = network_map.get(network)()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint {} ...'.format(checkpoint_path))
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/{}'.format(checkpoint_path))
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    net.eval()
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
        print('Accuracy of %5s : %2.1f %%' % (classes[i], 10 * class_correct[i] / class_total[i]))

def info():
    print('====================')
    print('| network:{}'.format(network))
    print('| dataset:{}'.format('cifar10'))
    print('| batch_size:{}'.format(test_bs))
    print('| dataset_worker:{}'.format(test_worker))
    print('| checkpoint path: {}'.format(checkpoint_path))

info()
test()
each_acc()

