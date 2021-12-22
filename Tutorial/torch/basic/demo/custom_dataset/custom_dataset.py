# -*- coding:utf-8 -*-

# Import
import torch
import os
import pandas as pd
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from torch.utils.data import (Dataset, DataLoader)
from torchvision import transforms
from torch import optim

# static parameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameter
batch_size = 64
num_epochs = 3
in_channel = 3
num_classes = 2
learning_rate = 1e-3


# load data
class CatAndDogDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotation = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotation.iloc[index, 0])
        y_label = self.annotation.iloc[index, 1]
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image, y_label


# load data
dataset = CatAndDogDataset(csv_file="cats_dogs.csv",
                           root_dir="cats_dogs_resized",
                           transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset=dataset, lengths=[5, 5])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

model = torchvision.models.googlenet(pretrained=False)
model.load_state_dict(torch.load(r"E:\DataSet\pretrained\googlenet\googlenet-1378be20.pth"))
model.to(device=device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=5, verbose=True)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
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

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
