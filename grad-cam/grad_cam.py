# %%
import sys
sys.path.append('/work/pytorch/grad-cam')

import os
import glob

import cv2
import timm
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torchvision.models import *
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

from utils import *
from visualisation.core import *
from visualisation.core.utils import device
from visualisation.core.utils import image_net_preprocessing, image_net_postprocessing

# %%
# 各カテゴリーから何枚の画像を処理するか
max_img = 5

# カテゴリー名の抽出
path = '../../datasets/stanford-dogs/'
interesting_categories = os.listdir(os.path.join(path, 'test'))

# 処理する画像をリスト化
images = [] 
for category_name in interesting_categories:
    image_paths = glob.glob(f'{path}/test/{category_name}/*')
    category_images = list(map(lambda x: PIL.Image.open(x), image_paths[:max_img]))
    images.extend(category_images)

# PyTorchで読み込める形式へ変換
inputs  = [Compose([Resize((224,224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in images]
inputs = [i.to(device) for i in inputs]

images = list(map(lambda x: cv2.resize(np.array(x),(224,224)),images))

# %%
# モデルの定義
# 重みなしでLoad
model = timm.create_model('efficientnet_b0')

# 最終のFC層を再定義
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(interesting_categories))

model = model.to(device)
model.eval()

# 学習済みの重みを定義
model_path = 'save_models/pytorch_test_albumentations/pytorch_test_albumentations_bs64_lr0.001_best.pkl'
model.load_state_dict(torch.load(model_path))

# %%
# Grad CAM
vis = GradCam(model, device)

grad_outs = list(map(lambda x: tensor2img(vis(x, None,postprocessing=image_net_postprocessing)[0]), inputs))
del model
torch.cuda.empty_cache()

# %%
# 画像をグリッド状に並べる
# NHWC -> NCHWの形式に変換
images = np.transpose(grad_outs, [0,3,1,2])
# PyTorchのテンソルにする
images_tensor = torch.as_tensor(images)

# グリッド状に並べる
grid_images_tensor = torchvision.utils.make_grid(images_tensor, 
                                                   nrow=5, # 1行あたりの画像数
                                                   padding=30 # 画像間の間隔
                                                   )

# PyTorchのテンソルをNumpy配列に変換し、NCHW -> NHWCの形式に変換
grid_images = np.transpose(grid_images_tensor.numpy(), [1,2,0])

def torchvision_plot():
    plt.axis('off')
    plt.imshow(grid_images)
    plt.show()

# %%
# グリッド画像の表示
torchvision_plot()

# グリッド画像の保存（make_gridを定義）
torchvision.utils.save_image(images_tensor, 
                             'grad-cam/grid_image.png',
                             nrow=5, 
                             padding=30)
