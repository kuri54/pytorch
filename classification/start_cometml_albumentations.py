# %%
# %load_ext autoreload
# %autoreload 2

os.chdir('../')

# %%
from comet_ml import Experiment, ConfusionMatrix

# %%
import os
import copy

import cv2
import timm
import numpy as np
import pandas as pd
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import *
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data

import torchvision
from torchvision import datasets, models, transforms

from classification.dataset import MyDataset, make_filepath_list
from classification.evaluator import plot_roc_fig
from classification.train import train_model_cometml, visualize_model_cometml

# %%
# loggingを開始 -> API情報を載せた'.comet.config'をhomeディレクトリに作成しているので、APIの入力は必要ない
project_name = 'pytorch_test_albumentations'
experiment = Experiment(project_name=project_name)

# ハイパラをlogging
hyper_params = {
    'num_classes': 5,
    'batch_size': 64,
    'num_epochs': 500,
    'learning_rate': 0.001
}

experiment.log_parameters(hyper_params)
experiment.set_name('batch_size: {} learning_rate: {}'.format(hyper_params['batch_size'], hyper_params['learning_rate']))

# %%
# 画像データへのファイルパスを格納したリストを取得する
data_dir_path = '../../datasets/stanford-dogs'

train_list = make_filepath_list(data_dir_path, phase='train')
valid_list = make_filepath_list(data_dir_path, phase='valid')
test_list = make_filepath_list(data_dir_path, phase='test')

# %%
# albumentationsでデータ水増しと正規化
train_transform_albu = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(224, 224),
    A.Cutout(p=0.5),
    A.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
    ToTensorV2()
    ])

valid_transform_albu = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
    ToTensorV2()
    ])

test_transform_albu = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
    ToTensorV2()
    ])


class_names = os.listdir(os.path.join(data_dir_path, 'train'))
class_names

train_dataset = MyDataset(train_list, class_names, transform=train_transform_albu)
valid_dataset = MyDataset(valid_list, class_names, transform=valid_transform_albu)
test_dataset = MyDataset(test_list, class_names, transform=test_transform_albu)

# %%
# シード値の固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

np.random.seed(42)

# dataloaderの作成
dataloaders = {'train': torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=hyper_params['batch_size'],
                                                    shuffle=True,
                                                    num_workers=2,
                                                    worker_init_fn=worker_init_fn),
               'valid': torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=hyper_params['batch_size'],
                                                    num_workers=2,
                                                    worker_init_fn=worker_init_fn),
               'test': torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=hyper_params['batch_size'],
                                                   num_workers=2,
                                                   worker_init_fn=worker_init_fn)}

# %%
# GPU使用設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %%
# dataloaderの中身を確認
def imshow(inp, title=None):
    """Imshow for Tensor."""
    plt.figure(figsize=(18,15))
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

imshow(out
       # , title=[class_names[x] for x in classes]
       )

# %%
# モデルの定義 -> GPUへ送る
# timmのpretrained modelを表示
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

# EfficientNet_B0
model_ft = timm.create_model('efficientnet_b0', pretrained=True)

# PreAct-resnet18
# model_ft = torch.hub.load('ecs-vlc/FMix:master', 'preact_resnet18_cifar10_baseline', pretrained=True)
model_ft

# 最終のFC層を再定義
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, hyper_params['num_classes'])

model_ft = model_ft.to(device)

experiment.set_model_graph(model_ft)

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

# %%
# cm用のミニ画像を作成（valid用）
def valid_index_to_example(index):
    tmp, _ = valid_dataset[index]
    img = tmp.numpy()[0]
    image_name = 'confusion-matrix-{}.png'.format(index)
    data = experiment.log_image(img, name=image_name)

    return {"sample": image_name, "assetId": data["imageId"]}

# cm用のミニ画像を作成（test用）
def test_index_to_example(index):
    tmp, _= test_dataset[index]
    img = tmp.numpy()[0]
    image_name = 'test-{}.png'.format(index)
    data = experiment.log_image(img, name=image_name)

    return {"sample": str(index), "assetId": data["imageId"]}

valid_confusion_matrix = ConfusionMatrix(labels=class_names,
                                         index_to_example_function=valid_index_to_example)
test_confusion_matrix = ConfusionMatrix(labels=class_names,
                                        index_to_example_function=test_index_to_example)

# %%
model = train_model_cometml(experiment,
                            hyper_params,
                            valid_confusion_matrix,
                            model_ft,
                            dataloaders,
                            class_names,
                            device,
                            criterion,
                            optimizer,
                            scheduler=cos_lr_scheduler,
                            save_model_name=project_name,
                            num_epochs=hyper_params['num_epochs']
                            )

visualize_model_cometml(experiment,
                        test_confusion_matrix,
                        model_ft,
                        dataloaders,
                        class_names,
                        device)

experiment.end()

# %%
