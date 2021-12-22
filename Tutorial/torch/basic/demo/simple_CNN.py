# -*- coding:utf-8 -*-
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn, true_divide
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1)
        )
        self.fc1 = nn.Linear(in_features=16 * 7 * 7, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# HyperParameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# load data
train_dataset = datasets.MNIST(r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\\',
                               download=True, transform=transforms.ToTensor(), train=True)
test_dataset = datasets.MNIST(r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\\',
                              download=True, transform=transforms.ToTensor(), train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimzer
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        target = target.to(device=device)
        # forward
        scores = model(data)
        loss = criterion(scores, target)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def check_accuracy(loader: DataLoader, model: nn.Module) -> float:
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return true_divide(num_correct, num_samples)


print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")
