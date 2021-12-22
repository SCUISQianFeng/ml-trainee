# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim, true_divide
from torch.utils.data import DataLoader
from tqdm import tqdm

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
# load data
train_dataset = torchvision.datasets.MNIST(root=r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\\',
                                           train=False, transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root=r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\\',
                                          train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=sequence_length * hidden_size, out_features=num_classes)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(size=(self.num_layers, x.size(0), self.hidden_size)).to(device)  # 2 * 64 * 256
        # forward propagate LSTM
        out, h1 = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=sequence_length * hidden_size, out_features=num_classes)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(size=(self.num_layers, x.size(0), self.hidden_size)).to(device)  # 2 * 64 * 256
        # forward propagate LSTM
        out, h1 = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True)
        self.fc = nn.Linear(in_features=sequence_length * hidden_size, out_features=num_classes)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(size=(self.num_layers, x.size(0), self.hidden_size)).to(device=device)  # 2 * 64 * 256
        c0 = torch.zeros(size=(self.num_layers, x.size(0), self.hidden_size)).to(device=device)
        # forward propagate LSTM
        out, (h1, c1) = self.rnn(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


model = RNN_LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)

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
check_accuracy(test_loader, model)
