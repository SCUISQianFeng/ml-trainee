# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim, true_divide
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# hyper parameters
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
sequence_length = 28
learning_rate = 0.005
batch_size = 64
num_epochs = 2
# static parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_transforms = transforms.Compose([
    transforms.Resize((36, 36)),  # Resizes (32,32) to (36,36)
    transforms.RandomCrop((32, 32)),  # Takes a random (32,32) crop
    transforms.ColorJitter(brightness=0.5),  # Change brightness of image
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])

# load data
train_dataset = torchvision.datasets.CIFAR10(root=r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\CIFAR\CIFAR10\\',
                                           train=True, transform=my_transforms, download=False)
# test_dataset = torchvision.datasets.MNIST(root=r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\CIFAR\CIFAR10\\',
#                                           train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# class Bi_LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(Bi_LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
#                            batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(in_features=2 * hidden_size, out_features=num_classes)
#
#     def forward(self, x):
#         # set initial hidden and cell states
#         h0 = torch.zeros(size=(self.num_layers * 2, x.size(0), self.hidden_size)).to(device=device)  # 2 * 64 * 256
#         c0 = torch.zeros(size=(self.num_layers * 2, x.size(0), self.hidden_size)).to(device=device)
#         # forward propagate LSTM
#         out, (h1, c1) = self.rnn(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# model = Bi_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
model = CNN(in_channels=3, num_classes=10)

# loss and optimizer
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# train network
for epoch in range(num_epochs):
    for batch_index, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device=device).squeeze(1)
        target = target.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = criterion(scores, target)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()


def check_accuracy(loader: DataLoader, model: nn.Module):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    print(
        f"Got {num_correct} / {num_samples} with  accuracy {true_divide(float(num_correct), float(num_samples)) * 100:.2f}")


check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)
