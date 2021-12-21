# -*- coding:utf-8 -*-

import sys

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import sys

sys.path.append('..')

sys.path.append('..')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 12
BATCH_SIZE = 5
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = 'b7.pth.tar'
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1

basic_transforms = A.Compose([
    A.Resize(height=448, width=448),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0),
    ToTensorV2()]
)
