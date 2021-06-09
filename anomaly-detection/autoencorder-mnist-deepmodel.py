'''
https://benjoe.medium.com/anomaly-detection-using-pytorch-autoencoder-and-mnist-31c5c2186329

このモデルではうまく学習できていない（Lossが下がりきっていない）ため、可視化がうまくいかない。
-> 単に層を深くするだけでは学習が収束しない？
'''
# %%
import time
import random
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# %%
df = pd.read_csv('./anomaly-detection/data/mnist_test.csv')

# index 0-999のデータを異常データとし、残りはそのままにしておく
anom = df[:1000]
clean = df[1000:]

# mnist_test.csvの最初の1000行(anom)を異常値として別のDataFrameに保存し、あとで結合する
for i in range(len(anom)):
    row = anom.iloc[i]
    # 行の各要素を繰り返し処理
    for i in range(len(row)-1):
        # 要素にノイズを加える
        row[i+1] = min(255, row[i+1]+random.randint(100,200))

# ノイズを加えるだけでなく、ラベルを「異常あり：1」と「異常なし：0」の2値にする
# 最終的にはこのラベルを使って、どれだけの異常を発見できたか判断する
anom['label'] = 1
clean['label'] = 0

# 2つのDataFrameを結合し、シャッフルして保存する
an_test = pd.concat([anom, clean])
an_test.sample(frac=1)
an_test.to_csv('./anomaly-detection/data/anom.csv', 
               index=False
               )

# %%
# モデルの定義
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU()
        )
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode

# %%
# パラメータ
batch_size = 32
lr = 1e-2
w_d = 1e-5
# momentum = 0.9
epochs = 15

# %%
# custom dataset loaderの作成 -> 各行のラベル列を削除、0-1の範囲に正規化
class Loader(torch.utils.data.Dataset):
    def __init__(self):
        super(Loader, self).__init__()
        self.dataset = ''
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        row = row.drop(labels={'label'}) # ラベル列を削除
        data = torch.from_numpy(np.array(row)/255).float() # 正規化してnumpy配列をpytorchテンソルへ変換
        return data
    
class Train_Loader(Loader):
    def __init__(self):
        super(Train_Loader, self).__init__()
        self.dataset = pd.read_csv(
                       './anomaly-detection/data/mnist_train.csv',
                       index_col=False
                       )

class Test_Loader(Loader):
    def __init__(self):
        super(Test_Loader, self).__init__()
        self.dataset = pd.read_csv(
                       './anomaly-detection/data/anom.csv',
                       index_col=False
                       )

train_set = Train_Loader()
train_ = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
            drop_last=True
        )

test_set = Test_Loader()
test_ = torch.utils.data.DataLoader(
            test_set,
            batch_size=len(test_set),
            num_workers=20,
            pin_memory=True,
            drop_last=True
        )

# %%
# trainの設定
# トレーニング中に複数の値を追跡するもの
metrics = defaultdict(list)

# GPU使用設定 -> モデルをGPUへ転送
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AE()
model.to(device)

# 損失関数とオプティマイザの定義
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

# 学習ループ
model.train()
start = time.time()
for epoch in range(epochs):
    ep_start = time.time()
    running_loss = 0.0
    
    for bx, (encoding_img) in enumerate(train_):
        decoded_img = model(encoding_img.to(device))
        loss = criterion(encoding_img.to(device), decoded_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    epoch_loss = running_loss/len(train_set)
    metrics['train_loss'].append(epoch_loss)
    ep_end = time.time()
    
    print('-----------------------------------------------')
    print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch+1, epochs, epoch_loss))
    print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
    
end = time.time()
print('-----------------------------------------------')
print('[System Complete: {}]'.format(timedelta(seconds=end-start)))

# %%
# lossをプロット
fig, ax = plt.subplots(1,1,figsize=(7,5))
ax.set_title('Loss')
ax.plot(metrics['train_loss'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.savefig('./anomaly-detection/save/loss_deepmodel.png')

# %%
# 出力と入力のlossを用いて、入力が異常かどうかを判断するモデル
# -> lossが大きいと、モデルは既知の分布表現から外れた要素を見ていると仮定する
# このことを実現するためにテストセットを使ってlossを保持する
model.eval()
loss_dist = []
for bx, encoding_img in enumerate(next(iter(test_))):
# for i in range(len(anom)):
    # encoding_img = torch.from_numpy(np.array(anom.iloc[i][1:])/255).float()
    decoded_img = model(encoding_img.to(device))
    loss = criterion(encoding_img.to(device), decoded_img)
    loss_dist.append(loss.item())

# %%
# lossを視覚化することでどこに異常が隠れているかの情報を得る
loss_sc = []
for i in loss_dist:
    loss_sc.append((i, i))

lower_threshold = 0.0
upper_threshold = 0.3

fig, ax = plt.subplots(figsize=(6, 8))
plt.title('Loss Distribution')
plt.axis('off')

ax1 = fig.add_subplot(2, 1, 1)
ax1.scatter(*zip(*loss_sc))
ax1.axvline(upper_threshold, 0.0, 1, color='r')
ax1.set_xlabel('Loss')
ax1.set_ylabel('Loss')

ax2 = fig.add_subplot(2, 1, 2)
sns.distplot(loss_dist, bins=100, kde=True, color='blue')
ax2.axvline(upper_threshold, 0.0, 10, color='r')
ax2.axvline(lower_threshold, 0.0, 10, color='b')
ax2.set_xlabel('Loss')
ax2.set_ylabel('Number of sumples')

fig.tight_layout()
plt.savefig('./anomaly-detection/save/Threshold.png')
fig.show()

'''プロット図
赤線を超えた（右にある）データは異常と判断できる（推定閾値）
青線は下限の閾値
'''

# %%
# 上記の閾値を用いて何を異常とみなすかを予測し、その発生数をカウントする
df = pd.read_csv('./anomaly-detection/data/anom.csv')
ddf = pd.DataFrame(columns=df.columns)

tp = 0
fp = 0
tn = 0
fn = 0
total_anom = 0
for i in range(len(loss_dist)):
    total_anom += df.iloc[i]['label']
    if loss_dist[i] >= upper_threshold: # 閾値の設定
        n_df = pd.DataFrame([df.iloc[i]])
        n_df['loss'] = loss_dist[i]
        ddf = pd.concat([df,n_df], sort = True)
        if float(df.iloc[i]['label']) == 1.0:
            tp += 1
        else:
            fp += 1
    else:
        if float(df.iloc[i]['label']) == 1.0:
            fn += 1
        else:
            tn += 1
print('[TP] {}\t[FP] {}\t[MISSED] {}'.format(tp, fp, total_anom-tp))
print('[TN] {}\t[FN] {}'.format(tn, fn))

# %%
# 混合同列をプロット
conf = [[tn,fp],[fn,tp]]
fig, ax = plt.subplots()
sns.heatmap(conf, 
            annot=True, 
            # annot_kws={"size": 10}, 
            fmt='d', 
            cmap='Blues'
            )
ax.set_ylim(len(conf), 0)
plt.savefig('./anomaly-detection/save/confusion_matrix.png')
plt.show()
