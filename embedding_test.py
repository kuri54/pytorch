# %%
from comet_ml import Experiment

import os
import glob
import albumentations as A
import cv2
from classification.dataset import MyDataset, make_filepath_list

# %%
# loggingを開始 -> API情報を載せた'.comet.config'をhomeディレクトリに作成しているので、APIの入力は必要ない
project_name = 'pytorch_test_embedding'
experiment = Experiment(project_name=project_name)

data_dir_path = '../../datasets/stanford-dogs'
train_list = make_filepath_list(data_dir_path, phase='train')
labels = os.listdir(os.path.join(data_dir_path, 'train'))

vectors = []
labels = []
for img in train_list:
    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (28, 28))
    vectors.append(im)
    
    names = img.split('/')
    label = names[-2]
    labels.append(label)

# %%
experiment.log_embedding(
    vectors,
    labels,
    # image_data=vectors,
    # image_size=(28, 28)
    )

experiment.end()
