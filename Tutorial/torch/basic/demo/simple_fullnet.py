# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, true_divide

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
batch_size = 64
learning_rate = 1e-3
epochs = 3
input_size = 784
num_classes = 10

# load data
train_data = datasets.MNIST(root=r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\\',
                            train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.MNIST(root=r'E:\DataSet\DataSet\ClassicalDatasets\MNIST\\',
                           train=False, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


# create model
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

# loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# train model

for epoch in range(epochs):
    for batch_index, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        target = target.to(device=device)
        # correct shape
        data = data.reshape(data.shape[0], -1)  # n x m

        # froward
        scores = model(data)
        loss = criterion(scores, target)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# valid model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # 用训练的model来计算得分

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)   # max(1)在类别结果列上返回每个类的对应的得分，即（得分，类别）
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)  # size(0) : batch_size

    model.train()
    return torch.true_divide(num_correct, num_samples)

print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")