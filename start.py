# %%
# import sys
# print(sys.path)
# sys.path.append('/work/pytorch/model')

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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=0, worker_init_fn=worker_init_fn)
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
# 学習ループ
def train_simple_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

# %%
# test画像で検証 -> 可視化
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            correct = 0
            total = 0
            correct += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                plt.subplots_adjust(wspace=0.4, hspace=1.0)
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Predicted: {}\nGround Truth: {}'.format(class_names[preds[j]], class_names[labels[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
                
        model.train(mode=was_training)

# %%
# 学習ループ：metricsとtensorboard出力あり
def train_binary_model_metrics(model, criterion, optimizer, scheduler, num_epochs=25, save_model_name='binary_model', save_tensorboard_name='binary_runs'):
    writer = SummaryWriter('tensorboard_runs/{}/{}'.format(save_tensorboard_name, save_model_name))
    save_model_dir = 'save_models/{}'.format(save_model_name)
    os.makedirs(save_model_dir, exist_ok=True)
    d = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    save_day = '{}_{}{}_{}-{}'.format(d.year, d.month, d.day, d.hour, d.minute)
    since = time.time()

    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0
    best_precision = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    axis = 1
                    _, preds = torch.max(outputs, axis)
                    loss = criterion(outputs, labels) 

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 学習の評価＆統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                if epoch%10 == 0:
                    torch.save(model_ft.state_dict(), os.path.join(save_model_dir, save_model_name+'_{}_{}.pkl'.format(epoch, save_day)))
                    print("saving model epoch :{}".format(epoch))

            # 評価項目 (loss, accracy, recall, precision, f1-score)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            metrics_dict = classification_report(y_true=labels.cpu(), y_pred=preds.cpu(), output_dict=True)
                        
            fpr, tpr, thresholds = roc_curve(y_true=labels.cpu(), y_score=preds.cpu())
            auc = roc_auc_score(y_true=labels.cpu(), y_score=preds.cpu())
            
            fig = plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc))
            plt.legend(loc='lower right')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            epoch_recall = metrics_dict['macro avg']['recall']
            epoch_precision = metrics_dict['macro avg']['precision']
            epoch_f1 = metrics_dict['macro avg']['f1-score']

            writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)
            writer.add_scalar('Recall/{}'.format(phase), epoch_recall, epoch)
            writer.add_scalar('Precision/{}'.format(phase), epoch_precision, epoch)
            writer.add_scalar('F1-score/{}'.format(phase), epoch_f1, epoch)
            writer.add_figure('ROC/{}'.format(phase), fig, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1-score: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                if epoch_recall==1 and epoch_precision > best_precision:
                    torch.save(model_ft.state_dict(), 
                               os.path.join(save_model_dir, save_model_name+'_{}_{}_recall_1.0.pkl'.format(epoch, save_day)))
                    print('saving model recall=1.0 epoch :{}'.format(epoch))
                    recall_1_precision = epoch_precision
                best_precision = epoch_precision
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, Precision: {:.4f}'.format(best_acc, best_precision))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model_ft.state_dict(), 
               os.path.join(save_model_dir, save_model_name+'_{}_{}_best.pkl'.format(epoch, save_day)))
    writer.close()

    return model

# %%
# 学習ループ：metricsとtensorboard出力あり
def train_multiple_model_metrics(model, criterion, optimizer, scheduler, num_epochs=25, save_model_name='multiple_model', save_tensorboard_name='multiple_runs'):
    writer = SummaryWriter('tensorboard_runs/{}/{}'.format(save_tensorboard_name, save_model_name))
    save_model_dir = 'save_models/{}'.format(save_model_name)
    os.makedirs(save_model_dir, exist_ok=True)
    d = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    save_day = '{}_{}{}_{}-{}'.format(d.year, d.month, d.day, d.hour, d.minute)
    since = time.time()

    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0
    best_precision = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    axis = 1
                    _, preds = torch.max(outputs, axis)
                    loss = criterion(outputs, labels) 
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # 学習の評価＆統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # confusion_matrixを描く
                # dataloaderバッチのインデックスを設定しておかないと端数でエラーが出る
                if idx == 2:
                    plot_image_array = plot_confusion_matrix(labels.cpu(), preds.cpu(), class_names)
            
            if phase == 'train':
                scheduler.step()
                if epoch%10 == 0:
                    torch.save(model_ft.state_dict(), os.path.join(save_model_dir, save_model_name+'_{}_{}.pkl'.format(epoch, save_day)))
                    print('saving model epoch :{}'.format(epoch))
                    
            # 評価項目 (loss, accracy, recall, precision, f1-score)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            metrics_dict = classification_report(y_true=labels.cpu(), y_pred=preds.cpu(), output_dict=True)

            epoch_recall = metrics_dict['macro avg']['recall']
            epoch_precision = metrics_dict['macro avg']['precision']
            epoch_f1 = metrics_dict['macro avg']['f1-score']
            end_epoch = len(range(num_epochs))
            
            writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)
            writer.add_scalar('Recall/{}'.format(phase), epoch_recall, epoch)
            writer.add_scalar('Precision/{}'.format(phase), epoch_precision, epoch)
            writer.add_scalar('F1-score/{}'.format(phase), epoch_f1, epoch)

            # Confusion Matrixは最終epochのもののみを保存
            writer.add_image('Confusion Matrix/{}'.format(phase), plot_image_array, end_epoch)

            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1-score: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_f1))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                if epoch_recall==1 and epoch_precision > best_precision:
                    torch.save(model_ft.state_dict(), 
                               os.path.join(save_model_dir, save_model_name+'_{}_{}_recall_1.0.pkl'.format(epoch, save_day)))
                    print('saving model recall=1.0 epoch :{}'.format(epoch))
                    recall_1_precision = epoch_precision
                best_precision = epoch_precision
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}, Precision: {:.4f}'.format(best_acc, best_precision))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model_ft.state_dict(), 
               os.path.join(save_model_dir, save_model_name+'_{}_{}_best.pkl'.format(epoch, save_day)))
    writer.close()
    
    return model

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
optimizer_ft = optim.SGD(model_ft.parameters(), 
                         lr=0.001,
                         momentum=0.9, # モーメンタム係数
                         nesterov=False # ネステロフ加速勾配法
                         )

# 学習スケジューラ
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%
# 学習
model_ft = train_simple_model(model_ft,  
                              criterion, 
                              optimizer_ft, 
                              exp_lr_scheduler,
                              num_epochs=2)

# %%
visualize_model(model_res)

# %%
# 学習
model_ft = train_multiple_model_metrics(model_ft, 
                                        criterion, 
                                        optimizer_ft, 
                                        exp_lr_scheduler, 
                                        num_epochs=2, 
                                        save_model_name='metrics_test_model_multiple_ft',
                                        save_tensorboard_name='image_test_runs')

# %%
visualize_model(model_ft)

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
