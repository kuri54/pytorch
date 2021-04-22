import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from sklearn.metrics import *

import torch.utils.data as data
from torchvision import transforms

def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    """
    datasetを3つのサブセットに分割する(test,validation,train)
    
    Parameters
    ----------
    data_dir_path: str
        データセットが格納されているディレクトリのパス
    train_ratio: int
        train用に分割する割合
    valid_ratio: int
        validatioｎ用に分割する割合
    
    Returns
    -------
    path_to_datasetの直上にそれぞれのフォルダが作成される
    """
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

    return

def plot_confusion_matrix(labels, preds, class_names):
    """
    Confusion Matrixをプロットする
    
    Parameters
    ----------
    labels: int
        正解ラベル（CPUに戻しておく必要がある）
    preds: int
        予測ラベル（CPUに戻しておく必要がある）
    class_names: list of str
        ラベル名のリスト
    
    Returns
    -------
    Confusion Matrixをプロットしたimageファイル
    """
    fig = plt.figure()
    plt.ion()
    plt.rcParams['figure.subplot.bottom'] = 0.2
    cm = confusion_matrix(y_true=labels, y_pred=preds)
    cm = pd.DataFrame(data=cm, index=class_names, columns=class_names)
    sns.heatmap(cm, annot=True, cmap='Blues', square=True)
    plt.ylim(0, cm.shape[0])
    plt.xlabel('Prediction')
    plt.ylabel('Label (Ground Truth)')
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.close(fig)

    # 画像がジャギらないようにimage形式に変換する
    canvas = fig.canvas.draw()
    plot_image = fig.canvas.renderer._renderer
    plot_image_array = np.array(plot_image).transpose(2, 0, 1)
    
    return plot_image_array

# class ImageTransform(object):
#     '''
#     入力画像の前処理をするクラス
#     train: 一定の確率で左右反転
#            ランダムにCropしてリサイズ
#            Tensor型に変換
#            標準化
# 
#     validation、test: リサイズ
#                       Tensor型に変換
#                       標準化
# 
#     Attributes
#     ----------
#     data_trasnform: dict
#         各処理をまとめておく
#     '''
# 
#     def __init__(self, resize):
#         '''
#         Parameters
#         ----------
#         resize: int
#             リサイズする画像の大きさ
#         '''
#         self.data_trasnform = {
#             'train': transforms.Compose([
#                 transforms.RandomResizedCrop(resize), 
#                 transforms.RandomHorizontalFlip(p=0.5), # 一定の確率（ｐ）で左右反転
#                 transforms.ToTensor(), # Tensor型に変換
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]) # 色情報の標準化
#             ]),
#             'validation': transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(resize),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
#             ]),
#             'test': transforms.Compose([
#                 transforms.Resize((resize, resize)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])])
#             }
# 
#     def __call__(self, image, phase='train'):
#         '''
#         Parameters
#         ----------
#         phase: str
#             データセットのフェーズ
# 
#         Returns
#         -------
#         data_trasnform[phase](image): object
#         '''
#         return self.data_trasnform[phase](image)
# 
# class MyDataset(data.Dataset):
#     '''
#     Datasetの作成クラス
#     PyTorchのDatasetクラスを継承させる
# 
#     Attrbutes
#     ---------
#     data_dir_path:
#         データセットが格納されているディレクトリのパス
#     classes: list of str
#         ラベル名のリスト
#     transform: object
#         前処理クラスのインスタンス
#     phase: str
#         各phaseで前処理が違うため、個別に処理するフラグ
#     data_phase: str
#         フェーズまでが記されているパス
#     image_paths: list of str
#         各画像のパスのリスト
#     '''
# 
#     def __init__(self, data_dir_path, classes, transform=None, phase='train'):
#         '''
#         Parameters
#         ----------
#         data_dir_path:
#             データセットが格納されているディレクトリのパス
#         classes: list of str
#             ラベル名のリスト
#         transform: object
#             前処理クラスのインスタンス
#         phase: str
#             各phaseで前処理が違うため、個別に処理するフラグ
#         '''
#         self.data_dir_path = data_dir_path
#         self.classes = classes
#         self.transform = transform
#         self.phase = phase
# 
#         self.data_phase = os.path.join(self.data_dir_path, self.phase)
#         self.image_paths = [str(path) for path in Path(self.data_phase).glob("**/*.jpg")]
# 
#     def __len__(self):
#         '''
#         Returns
#         -------
#         画像の枚数を返す
#         '''
#         return len(self.image_paths)
# 
#     def __getitem__(self, index):
#         '''
#         Parameters
#         ----------
#         index: int
#             各画像のインデックス
# 
#         Returns
#         -------
#         img_transformed: 
#             前処理後の画像
#         label: int
#             インデックスに変換した画像ラベル
#         '''
#         path = self.image_paths[index]
#         image = Image.open(path)
# 
#         img_transformed = self.transform(image, self.phase) # phaseに応じた前処理
# 
#         label = self.image_paths[index].split("/")[5] # ラベル情報を取得
#         label = self.classes.index(label) # 取得したラベル名をインデックスに変換
# 
#         return img_transformed, label
