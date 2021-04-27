from comet_ml import Experiment

# %%
import os
import time
import copy
import datetime
import pytz

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, models, transforms

# %%
hyper_params = {
    'image_size': 28,
    'num_classes': 5,
    'batch_size': 128,
    'num_epochs': 20,
    'learning_rate': 0.01
}

# experiment.log_parameters(hyper_params)

# MNIST Dataset
train_dataset = datasets.CIFAR10(root='/datasets/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

test_dataset = datasets.CIFAR10(root='/datasets/',
                                train=False,
                                transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=hyper_params['batch_size'],
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=hyper_params['batch_size'],
                                          shuffle=False)

dataloaders = {
    'train': train_loader,
    'valid': test_loader
    }

class_names = train_dataset.classes
class_names

# %%
# GPU使用設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %%
# dataloaderの中身を確認
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.2, 0.2, 0.2])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.axis('off')
    plt.imshow(inp)
    # if title is not None:
    #     plt.title(title)

inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# %%
# モデルの定義 -> GPUへ送る
# ResNet18
model_ft = models.resnet18(pretrained=True)
model_ft

# 最終のFC層を再定義
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, hyper_params['num_classes'])

model_ft = model_ft.to(device)

# %%
# 損失関数
criterion = nn.CrossEntropyLoss()

# オプティマイザ
# モーメンタム付きSGDが割と最強らしい
optimizer = optim.SGD(model_ft.parameters(), 
                         lr=hyper_params['learning_rate'],
                         momentum=0.9, # モーメンタム係数
                         nesterov=False # ネステロフ加速勾配法
                         )

# 学習スケジューラ
# とりあえずCosineAnnealingLRを使っとけ
cos_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=hyper_params['learning_rate'])

# Train the Model

# %%
# 学習ループ
def train_simple_model(model, dataloaders, class_names, device, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            experiment.log_metric('{}_loss'.format(phase), running_loss / len(dataloaders[phase].dataset), step=epoch)
            experiment.log_metric('{}_acc'.format(phase), running_corrects.double() / len(dataloaders[phase].dataset), step=epoch)
    
        print()
    
    print('Fin')

    return model

experiment = Experiment(api_key='GpKpecp9yQMMBwP1cWQOSZdFT', project_name="pytorch_test2")

with experiment.train():
    train_simple_model(model_ft, 
                       dataloaders, 
                       class_names, 
                       device, 
                       criterion, 
                       optimizer, 
                       scheduler=cos_lr_scheduler, 
                       num_epochs=hyper_params['num_epochs']
                       )
    
    
experiment.end()
