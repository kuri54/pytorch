# %%
# %load_ext autoreload
# %autoreload 2

os.chdir('../')
# %%
from comet_ml import Experiment, ConfusionMatrix

# %%
import os
import copy
import random

import timm
import numpy as np
from pprint import pprint
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import *
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data

import torchvision
from torchvision import datasets, models, transforms

from utils.evaluator import plot_roc_fig
from utils.criterion import LabelSmoothLoss
from utils.dataset import MyDataset, make_filepath_list
from utils.train import train_loop, valid_loop, evalute_model

# %%
# timmのpretrained modelを表示
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

# %%
# loggingを開始 -> API情報を載せた'.comet.config'をhomeディレクトリに作成しているので、APIの入力は必要ない
project_name = 'pytorch_test_albumentations'
experiment = Experiment(project_name=project_name)

# ハイパラをlogging
hyper_params = {
    'num_classes': 5,
    'batch_size': 64,
    'num_epochs': 500,
    'learning_rate': 0.001,
    'model_name': 'efficientnet_b0',
    'weight_decay': 5e-5,
    'seed': 64,
    'img_size': 224
}

experiment.log_parameters(hyper_params)
experiment.set_name('batch_size: {} learning_rate: {}'.format(hyper_params['batch_size'], hyper_params['learning_rate']))

# シードの設定
def seed_everything(seed:int==64):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(hyper_params['seed'])

# %%
# 画像データへのファイルパスを格納したリストを取得する
data_dir_path = '../../datasets/stanford-dogs'

train_list = make_filepath_list(data_dir_path, phase='train')
valid_list = make_filepath_list(data_dir_path, phase='valid')
test_list = make_filepath_list(data_dir_path, phase='test')

# %%
# albumentationsでデータ水増しと正規化
train_transform_albu = A.Compose([
    A.Resize(hyper_params['img_size'], hyper_params['img_size']),
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(224, 224),
    A.Cutout(p=0.5),
    A.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
    ToTensorV2()
    ])

valid_transform_albu = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(hyper_params['img_size'], hyper_params['img_size']),
    A.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
    ToTensorV2()
    ])

test_transform_albu = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(hyper_params['img_size'], hyper_params['img_size']),
    A.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
    ToTensorV2()
    ])


class_names = os.listdir(os.path.join(data_dir_path, 'train'))
class_names

train_dataset = MyDataset(train_list, class_names, transform=train_transform_albu)
valid_dataset = MyDataset(valid_list, class_names, transform=valid_transform_albu)
test_dataset = MyDataset(test_list, class_names, transform=test_transform_albu)

# %%
# dataloaderの作成
dataloaders = {'train': torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=hyper_params['batch_size'],
                                                    shuffle=True,
                                                    num_workers=2),
               'valid': torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=hyper_params['batch_size'],
                                                    num_workers=2),
               'test': torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=hyper_params['batch_size'],
                                                   num_workers=2)}

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
model_ft = timm.create_model(hyper_params['model_name'], pretrained=True,
                             num_classes=hyper_params['num_classes']
                             )

# PreAct-resnet18
# model_ft = torch.hub.load('ecs-vlc/FMix:master', 'preact_resnet18_cifar10_baseline', pretrained=True)
model_ft

model_ft = model_ft.to(device)

experiment.set_model_graph(model_ft)

# %%
# 損失関数
# criterion = nn.CrossEntropyLoss()

# Label Smoothing
criterion = LabelSmoothLoss(num_classes=hyper_params['num_classes'],
                            alpha=0.05
                            )

# オプティマイザ
# モーメンタム付きSGDが割と最強らしい
optimizer = optim.SGD(model_ft.parameters(),
                         lr=hyper_params['learning_rate'],
                         momentum=0.9, # モーメンタム係数
                         nesterov=False # ネステロフ加速勾配法
                         )

# optimizer = optim.AdamW(model_ft.parameters(),
#                         lr=hyper_params['learning_rate'],
#                         weight_decay=hyper_params['weight_decay'])

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
best_model_wts = copy.deepcopy(model_ft.state_dict())
best_acc = 0.0
best_loss = 1e+10
earlystop_counter = 0

task = 'binary'

if len(class_names) == 2:
    print('Task: Binary Class')
    task = 'binary'
else:
    print('Task: Multi Class')
    task = 'multi'

for epoch in range(hyper_params['num_epochs']):
    print('Epoch {}/{}'.format(epoch + 1, hyper_params['num_epochs']))
    print('-' * 10)

    train_loss, train_acc = train_loop(model_ft,
                                       dataloaders['train'],
                                       device,
                                       criterion,
                                       optimizer,
                                       scheduler=cos_lr_scheduler)

    experiment.log_metric('train_loss', train_loss, step=epoch)
    experiment.log_metric('train_acc', train_acc, step=epoch)

    valid_loss, valid_acc, labels_all, pred_all = valid_loop(model_ft,
                                                             dataloaders['valid'],
                                                             device,
                                                             criterion)

    metrics_dict = classification_report(y_true=labels_all, y_pred=pred_all, output_dict=True, zero_division=0)

    fig = plot_roc_fig(labels_all, pred_all, class_names, task)
    experiment.log_figure('ROC', fig, step=epoch)

    valid_recall = metrics_dict['macro avg']['recall']
    valid_precision = metrics_dict['macro avg']['precision']
    valid_f1 = metrics_dict['macro avg']['f1-score']

    experiment.log_metric('valid_loss', valid_loss, step=epoch)
    experiment.log_metric('valid_acc', valid_acc, step=epoch)

    valid_confusion_matrix.compute_matrix(labels_all, pred_all)
    experiment.log_confusion_matrix(matrix=valid_confusion_matrix,
                                    title='Confusion Matrix, Epoch #{}'.format(epoch + 1),
                                    file_name='confusion-matrix-{}.json'.format(epoch + 1)
                                    )

    print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
    print('Valid Loss: {:.4f} Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1-score: {:.4f}'.format(valid_loss, valid_acc, valid_recall, valid_precision, valid_f1))
    print()

    if valid_acc > best_acc:
        best_acc = valid_acc
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        print ('Save Best Acc Weight!')
        print()

    # early stop
    if valid_loss < best_loss:
        earlystop_counter = 0
        best_loss = valid_loss

    else:
        earlystop_counter += 1

    if earlystop_counter > 10:
        print()
        print('Early Stop!!!!')
        break

save_model_dir = './classification/save_models/{}'.format(project_name)
os.makedirs(save_model_dir, exist_ok=True)

model_ft.load_state_dict(best_model_wts)
torch.save(model_ft.state_dict(),
            os.path.join(save_model_dir, project_name+'_bs{}_lr{}_best.pkl'.format(hyper_params['batch_size'], hyper_params['learning_rate'])))
experiment.log_model('best_model',
                        os.path.join(save_model_dir, project_name+'_bs{}_lr{}_best.pkl'.format(hyper_params['batch_size'], hyper_params['learning_rate'])))

experiment.log_metric('best_val_acc', best_acc)

print('-' * 10)
print('Best val Acc: {:4f}'.format(best_acc))
print('Fin')

# %%
correct, total, labels_all, pred_all = evalute_model(model_ft,
                                                     dataloaders['test'],
                                                     device)

print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
experiment.log_metric('test_acc', 100. * correct / total)

test_confusion_matrix.compute_matrix(labels_all, pred_all)
experiment.log_confusion_matrix(matrix=test_confusion_matrix,
                                title='Test Confusion Matrix',
                                file_name='test-confusion-matrix.json'
                                )
# %%
experiment.end()

# %%
