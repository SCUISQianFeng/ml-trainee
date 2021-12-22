# -*- coding:utf-8 -*-


# Imports
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import torchvision
import torch
import torch.nn as nn
from torch import optim

# static parameter
file_name = r'E:\DataSet\pretrained\train_weight\my_checkpoint.pth.tar'


def save_checkpoint(state, filename=file_name):
    """
    保存模型
    :param state: 待保存的对象
    :param filename: 文件对象
    :return:
    """
    print('=> Saving checkpoint')
    torch.save(obj=state, f=filename)


def load_checkpoint(checkpoint, model: nn.Module, optimizer: optim):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def main():
    # Initialize network
    file_path = r'E:\DataSet\pretrained\vgg\vgg16-397923af.pth'
    model = torchvision.models.vgg16(pretrained=False)
    model.load_state_dict(torch.load(f=file_path))
    optimizer = optim.Adam(params=model.parameters(), )
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    save_checkpoint(checkpoint)

    load_checkpoint(torch.load(file_name), model, optimizer)


if __name__ == "__main__":
    main()
