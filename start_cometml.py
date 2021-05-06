# %%
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

# 関数移行後に削除
from sklearn.preprocessing import label_binarize
from itertools import cycle

from evaluator import plot_roc_fig
from password_api.my_api import *

# %%
# loggingを開始
experiment = Experiment(api_key=MY_COMETML_API_KEY, project_name="pytorch_test2")

# %%
# ハイパラをlogging
hyper_params = {
    'image_size': 32,
    'num_classes': 5,
    'batch_size': 128,
    'num_epochs': 2,
    'learning_rate': 0.01
}

experiment.log_parameters(hyper_params)

# %%
# data_path = '../../datasets/stanford-dogs/Test_Images'
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
# シード値の固定
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

np.random.seed(42)

# %%
# dataloaderの作成
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir_path, x),
                                          data_transforms[x])
                  for x in ['train', 'valid' , 'test']}
                  # for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=128,
                                              shuffle=True, 
                                              num_workers=2, 
                                              worker_init_fn=worker_init_fn)
              for x in ['train', 'valid', 'test']}
              # for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) 
                 for x in ['train', 'valid', 'test']}
                 # for x in ['train', 'valid']}
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

# cm用のミニ画像を作成（valid用）
def valid_index_to_example(index):
    tmp, _ = image_datasets['valid'][index]
    img = tmp.numpy()[0]
    data = experiment.log_image(img, name="valid_%d.png" % index)

    if data is None:
        return None

    return {"sample": str(index), "assetId": data["imageId"]}

# cm用のミニ画像を作成（test用）
def test_index_to_example(index):
    tmp, _ = image_datasets['test'][index]
    img = tmp.numpy()[0]
    data = experiment.log_image(img, name="test_%d.png" % index)

    if data is None:
        return None

    return {"sample": str(index), "assetId": data["imageId"]}

# %%
# 学習ループ（comet.mlへ転送）
def train_model_cometml(model, dataloaders, class_names, device, criterion, optimizer, scheduler, num_epochs=25):
    
    task = 'binary'
    
    if len(class_names) == 2:
        print('Task: Binary Class')
        task = 'binary'
    else:
        print('Task: Multi Class')
        task = 'multi'
    
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
            
            labels_all = []
            pred_all = []

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
                
                pred_all.extend(predict.item() for predict in preds)
                labels_all.extend(label.item() for label in labels)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            experiment.log_metric('{}_loss'.format(phase), running_loss / len(dataloaders[phase].dataset), step=epoch)
            experiment.log_metric('{}_acc'.format(phase), running_corrects.double() / len(dataloaders[phase].dataset), step=epoch)
            
            fig = plot_roc_fig(labels_all, pred_all, class_names, task)
            experiment.log_figure('ROC', fig, step=epoch)
            
            if phase == 'valid':
                experiment.log_confusion_matrix(labels_all, pred_all,
                                                labels=class_names, 
                                                title='Confusion Matrix, Epoch #{}'.format(epoch),
                                                file_name='confusion-matrix-{}.json'.format(epoch), 
                                                index_to_example_function=valid_index_to_example
                                                )
        
        print()
    
    print('Fin')


# %%
with experiment.train():
    train_model_cometml(model_ft,
                       dataloaders, 
                       class_names, 
                       device, 
                       criterion, 
                       optimizer, 
                       scheduler=cos_lr_scheduler, 
                       num_epochs=hyper_params['num_epochs']
                       )

'''
test用のループも作りたい
with experiment.test():
'''

experiment.end()
    
    
    
    
