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

# %%
# loggingを開始 -> API情報を載せた'.comet.config'をhomeディレクトリに作成しているので、APIの入力は必要ない
project_name = 'pytorch_test2'
experiment = Experiment(project_name=project_name)

# ハイパラをlogging
hyper_params = {
    'num_classes': 5,
    'batch_size': 256,
    'num_epochs': 100,
    'learning_rate': 0.0001
}

experiment.log_parameters(hyper_params)
experiment.set_name('batch_size: {} learning_rate: {}'.format(hyper_params['batch_size'], hyper_params['learning_rate']))

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

dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], 
                                              batch_size=hyper_params['batch_size'],
                                              shuffle=True, 
                                              num_workers=2, 
                                              worker_init_fn=worker_init_fn), 
               'valid': torch.utils.data.DataLoader(image_datasets['valid'], 
                                                             batch_size=hyper_params['batch_size'], 
                                                             num_workers=2, 
                                                             worker_init_fn=worker_init_fn),
               'test': torch.utils.data.DataLoader(image_datasets['test'], 
                                                             batch_size=hyper_params['batch_size'], 
                                                             num_workers=2, 
                                                             worker_init_fn=worker_init_fn)}

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

imshow(out
       # , title=[class_names[x] for x in classes]
       )

# %%
# モデルの定義 -> GPUへ送る
# ResNet18
model_ft = models.resnet18(pretrained=True)

# 最終のFC層を再定義
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, hyper_params['num_classes'])

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
    tmp, _ = image_datasets['valid'][index]
    img = tmp.numpy()[0]
    image_name = 'confusion-matrix-{}.png'.format(index)
    data = experiment.log_image(img, name=image_name)

    return {"sample": image_name, "assetId": data["imageId"]}

# cm用のミニ画像を作成（test用）
def test_index_to_example(index):
    tmp, _= image_datasets['test'][index]
    img = tmp.numpy()[0]
    image_name = 'test-{}.png'.format(index)
    data = experiment.log_image(img, name=image_name)

    return {"sample": str(index), "assetId": data["imageId"]}

# 学習ループ（comet.mlへ転送）
def train_model_cometml(model, dataloaders, class_names, device, criterion, optimizer, scheduler, num_epochs=25, save_model_name=project_name):
    save_model_dir = 'save_models/{}'.format(save_model_name)
    os.makedirs(save_model_dir, exist_ok=True)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
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
                experiment.train()
            else:
                model.eval()
                experiment.validate()

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
            
            metrics_dict = classification_report(y_true=labels_all, y_pred=pred_all, output_dict=True, zero_division=0)

            epoch_recall = metrics_dict['macro avg']['recall']
            epoch_precision = metrics_dict['macro avg']['precision']
            epoch_f1 = metrics_dict['macro avg']['f1-score']

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1-score: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1))
            
            experiment.log_metric('{}_loss'.format(phase), epoch_loss, step=epoch)
            experiment.log_metric('{}_acc'.format(phase), epoch_acc, step=epoch)
            experiment.log_metric('{}_recall'.format(phase), epoch_recall, step=epoch)
            experiment.log_metric('{}_precision'.format(phase), epoch_precision, step=epoch)
            experiment.log_metric('{}_f1'.format(phase), epoch_f1, step=epoch)
            
            fig = plot_roc_fig(labels_all, pred_all, class_names, task)
            experiment.log_figure('ROC', fig, step=epoch)
            
            if phase == 'valid':
                experiment.log_confusion_matrix(labels_all, pred_all,
                                                labels=class_names, 
                                                title='Confusion Matrix, Epoch #{}'.format(epoch + 1),
                                                file_name='confusion-matrix-{}.json'.format(epoch + 1), 
                                                index_to_example_function=valid_index_to_example
                                                )
        
            # best modelの保存
            if phase == 'valid' and epoch_acc > best_acc:
                # if epoch_recall==1 and epoch_precision > best_precision:
                #     torch.save(model.state_dict(), 
                #                os.path.join(save_model_dir, save_model_name+'_{}_{}_recall_1.0.pkl'.format(epoch)))
                #     print('saving model epoch :{}'.format(epoch))
                #     recall_1_precision = epoch_precision
                best_precision = epoch_precision
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 
               os.path.join(save_model_dir, save_model_name+'_bs{}_lr{}_best.pkl'.format(hyper_params['batch_size'], hyper_params['learning_rate'])))
    experiment.log_model('best_model',
                          os.path.join(save_model_dir,save_model_name+'_bs{}_lr{}_best.pkl'.format(hyper_params['batch_size'], hyper_params['learning_rate'])))
    
    experiment.log_metric('best_val_acc', best_acc)
    
    print('-' * 10)
    print('Best val Acc: {:4f}, Precision: {:.4f}'.format(best_acc, best_precision))
    print('Fin')

    return model

# test画像で検証 -> 可視化（comet.mlへ転送）
def visualize_model(model, dataloaders, class_names, device):
    experiment.test()
    model.eval() 

    with torch.no_grad():
        correct = 0
        total = 0
           
        labels_all = []
        pred_all = []
           
        for idx, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            pred_all.extend(predict.item() for predict in preds)
            labels_all.extend(label.item() for label in labels)
            
        print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
        experiment.log_metric('test_acc', 100. * correct / total)

    experiment.log_confusion_matrix(labels_all, pred_all,
                                    labels=class_names, 
                                    title='Test Confusion Matrix',
                                    file_name='test-confusion-matrix.json', 
                                    index_to_example_function=test_index_to_example
                                    )

# %%
model = train_model_cometml(model_ft,
                   dataloaders, 
                   class_names, 
                   device, 
                   criterion, 
                   optimizer, 
                   scheduler=cos_lr_scheduler, 
                   num_epochs=hyper_params['num_epochs']
                   )

visualize_model(model_ft, 
                dataloaders, 
                class_names, 
                device)

experiment.end()
