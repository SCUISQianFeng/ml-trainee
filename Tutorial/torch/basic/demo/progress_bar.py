# -*- coding:utf-8 -*-


from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

x = torch.randn(size=(1000, 3, 224, 224))
y = torch.randint(low=0, high=10, size=(1000, 1))
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=8)

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1, stride=1),
    nn.Flatten(),
    nn.Linear(10 * 224 * 224, 10),
)

NUM_EPOCHS = 100
for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader)
    for idx, (x, y) in enumerate(loop):
        scores = model(x)
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item())

