import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from itertools import cycle

from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

import torch.utils.data as data
from torchvision import transforms


def plot_cm(labels, preds, class_names):
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
    sns.heatmap(cm, annot=True, cmap='Blues', square=True, fmt='d')
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

def plot_roc(labels, preds, class_names, task):
    """
    タスク別にROC curveをプロットする
    
    Parameters
    ----------
    labels: int
        正解ラベル（CPUに戻しておく必要がある）
    preds: int
        予測ラベル（CPUに戻しておく必要がある）
    class_names: list of str
        ラベル名のリスト
    task: str
        タスクの指定（2値分類 or 多クラス分類）
    
    Returns
    -------
    ROC curveをプロットしたimageファイル
    """
    # 2値分類
    if task == 'binary':
        fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=preds)
        auc_binary = roc_auc_score(y_true=labels, y_score=preds)
        fig = plt.figure()
        plt.ion()
        plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc_binary))
        plt.title('ROC Curve Binary-class')
    
    # 多クラス分類
    elif task == 'multi':
        labels_list = list(range(len(class_names)))
        labels_binary = label_binarize(labels, classes=labels_list)
        pred_binary = label_binarize(preds, classes=labels_list)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(labels_binary[:, i], pred_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig = plt.figure()
        plt.ion()
        colors = cycle(['blue', 'red', 'green'])
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, 
                     label='ROC curve of {0} (area = {1:0.2f})' ''.format(class_names[i], roc_auc[i]))
        plt.title('ROC Curve Multi-class')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right") 
    plt.close(fig)
    
    # 画像がジャギらないようにimage形式に変換する
    canvas = fig.canvas.draw()
    plot_image = fig.canvas.renderer._renderer
    plot_image_roc = np.array(plot_image).transpose(2, 0, 1)
    
    return plot_image_roc

def plot_roc_fig(labels, preds, class_names, task):
    """
    タスク別にROC curveをfigure形式でプロットする
    
    Parameters
    ----------
    labels: int
        正解ラベル（CPUに戻しておく必要がある）
    preds: int
        予測ラベル（CPUに戻しておく必要がある）
    class_names: list of str
        ラベル名のリスト
    task: str
        タスクの指定（2値分類 or 多クラス分類）
    
    Returns
    -------
    ROC curveをプロットしたfigureファイル
    """
    # 2値分類
    if task == 'binary':
        fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=preds)
        auc_binary = roc_auc_score(y_true=labels, y_score=preds)
        fig = plt.figure()
        plt.ion()
        plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc_binary))
    
    # 多クラス分類
    elif task == 'multi':
        labels_list = list(range(len(class_names)))
        labels_binary = label_binarize(labels, classes=labels_list)
        pred_binary = label_binarize(preds, classes=labels_list)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(labels_binary[:, i], pred_binary[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig = plt.figure()
        plt.ion()
        colors = cycle(['blue', 'red', 'green'])
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, 
                     label='ROC curve of {0} (area = {1:0.2f})' ''.format(class_names[i], roc_auc[i]))
    
    plt.title('ROC Curve {}-class'.format(task))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right") 
    plt.close(fig)
        
    return fig
