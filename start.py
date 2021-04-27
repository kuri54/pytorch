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

from dataset import split_dataset_into_3, plot_confusion_matrix
from model.preact_resnet import PreActResNet18, PreActResNet50
from train import train_simple_model, visualize_model
from train import train_binary_model_metrics, train_multiple_model_metrics


# %%
data_path = '../../datasets/stanford-dogs/Test_Images'
data_dir_path = '../../datasets/stanford-dogs/'
# data_dir_path = '../../datasets/ants_bees/'

# %%
# データセットの分割
split_dataset_into_3(data_path, 0.7, 0.2)

# %%
# データ水増しと正規化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        # transforms.RandomCrop(size=(224,224), padding=4),
        # 画像の四方をパディングしたあとに、sizeでクロップ
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
    ])
    }

# %%
# シード値を固定
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

np.random.seed(42)

# %%
# dataloaderの作成
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir_path, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                             shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
              for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

print(dataset_sizes)
print(class_names)

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

# 最終のFC層を再定義
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

# %%
# PreActResNet18
model_res = PreActResNet18(in_channels=3, num_classes=5)
model_res

# Liner部分のサイズが合わないので再構築
model_res.linear = nn.Linear(512 * 7 * 7, 5)
model_res

model_res = model_res.to(device)

# %% 必要に応じて
# モデルのアーキテクチャをtensorboardに表示
writer = SummaryWriter('tensorboard_runs/metrics_test_runs/architecture_model_ft'))
writer.add_graph(model_ft, torch.rand(inputs.shape[0:]).to(device))
writer.close()

# %%
# 損失関数
criterion = nn.CrossEntropyLoss()

# オプティマイザ
# モーメンタム付きSGDが割と最強らしい
optimizer = optim.SGD(model_res.parameters(), 
                         lr=0.001,
                         momentum=0.9, # モーメンタム係数
                         nesterov=False # ネステロフ加速勾配法
                         )

# 学習スケジューラ
# とりあえずCosineAnnealingLRを使っとけ
cos_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# %%
# 学習
model_ft = train_simple_model(model_res,
                              dataloaders,
                              class_names, 
                              device,   
                              criterion, 
                              optimizer, 
                              scheduler=cos_lr_scheduler,
                              num_epochs=200)

# %%
visualize_model(model_ft, 
                dataloaders,
                class_names,
                device, 
                imshow, 
                num_images=6)

# %%
# 学習
model_res = train_multiple_model_metrics(model_res, 
                                        dataloaders,
                                        class_names, 
                                        device,   
                                        criterion, 
                                        optimizer, 
                                        scheduler=cos_lr_scheduler, 
                                        num_epochs=200, 
                                        save_model_name='model_multiple_res',
                                        save_tensorboard_name='res_runs')

# %%
visualize_model(model_res, 
                dataloaders,
                class_names,
                device, 
                imshow, 
                num_images=6)

# %%
# 最終層を除くすべてのネットワークをフリーズ
# backward()でグラデーションが計算されないようにする
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

# %%
visualize_model(model_conv)


# %%
# 学習
