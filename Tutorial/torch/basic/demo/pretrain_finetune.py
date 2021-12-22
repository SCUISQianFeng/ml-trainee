# -*- coding:utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, true_divide
# from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
batch_size = 64
learning_rate = 1e-3
batch_size = 1024
num_epochs = 3
num_classes = 10


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# load data
train_dataset = datasets.CIFAR10(root=r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\CIFAR\CIFAR10\\',
                                 train=True, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


model = torchvision.models.vgg16(pretrained=False, progress=True)
pretrain_model_path = r'E:\DataSet\pretrained\vgg\vgg16-397923af.pth'
model.load_state_dict(torch.load(pretrain_model_path))

for param in model.parameters():
    param.requires_grad = False

print(model)
# summary(model, (3, 224, 224))
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )

model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(in_features=512, out_features=100),
    nn.ReLU(),
    nn.Linear(100, num_classes)
)
model.to(device=device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    losses = []
    for batch_index, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        target = target.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        losses.append(loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses):.5f}")


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    for x, y in tqdm(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(
        f"Got {num_correct} / {num_samples} with accuracy {true_divide(float(num_correct), float(num_samples)) * 100:.2f}")


check_accuracy(train_loader, model)
