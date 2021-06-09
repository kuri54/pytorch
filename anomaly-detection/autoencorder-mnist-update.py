# %%
import os
import pylab
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64))

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Tanh() # [-1,1]に正規化しているのでTanhを用いる
        )

    def forward(self, x):
        encoding_img = self.encoder(x)
        decoded_img = self.decoder(encoding_img)
        return decoded_img

# %%
save_dir = './anomaly-detection/save'
data_dir = './anomaly-detection/data'

batch_size = 32
num_epochs = 100
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Autoencoder()
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)

# %%
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))  # MNISTは ToTensor()すると[0, 1] になる
                                            # 0.5を引いて0.5で割って [-1, 1] の範囲に変換する
])

train_dataset = MNIST(data_dir, download=True, train=True, transform=img_transform)
test_dataset = MNIST(data_dir, train=False, download=True, transform=img_transform)

# 特定の画像のみを抽出するため、datasetのデータを上書きする
# mask = (train_dataset.targets == 0) | (train_dataset.targets == 6)
# mask = (train_dataset.targets != 0)
mask = (train_dataset.targets == 1) # 1の画像のみをtrainに使用
train_dataset.data = train_dataset.data[mask]
train_dataset.targets = train_dataset.targets[mask]

train_1_loader = DataLoader(train_dataset, batch_size, shuffle=True)

mask = (test_dataset.targets == 9) | (test_dataset.targets == 1) # 9（異常）と1（正常）の画像のみにする
test_dataset.data = test_dataset.data[mask]
test_dataset.targets = test_dataset.targets[mask]

test_9_1_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# ラベルが制限されていることを確認
# for data, label in loader:
#     print(data.shape)
#     print(label)
#     break


# def to_img(x):
#     x = 0.5 * (x + 1)  # [-1,1] => [0, 1]
#     x = x.clamp(0, 1)
#     x = x.view(x.size(0), 1, 28, 28)
#     return x

# %%
model.train()
losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for img, _ in train_1_loader:
        # print("now")
        img = img.to(device)
        
        optimizer.zero_grad()
        
        input_img = img.view(img.size(0), -1)
        decoded_img = model(input_img)

        # 出力画像（再構成画像）と入力画像の間でlossを計算
        loss = criterion(decoded_img, input_img)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)

    print('epoch [{}/{}], loss: {}'.format(
        epoch + 1,
        num_epochs,
        epoch_loss))
    
    # 10エポックごとに再構成された画像（decoded_img）を描画する
    # if epoch % 10 == 0:
    #     pic = to_img(decoded_img.cpu().data)
    #     save_image(pic, save_dir+'/save/image_{}.png'.format(epoch))

# %%
# lossをプロット
fig, ax = plt.subplots(1,1,figsize=(7,5))
ax.set_title('Loss')
ax.plot(losses)
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.savefig('{}/loss.png'.format(save_dir))

# %%
images, labels = next(iter(test_9_1_loader))

model.eval()
loss_dist = []
test_imgs = []
decoded_imgs = []
for img in images:
    img = img.to(device)
    test_img = img.view(img.size(0), -1)

    decoded_img = model(test_img[0])
    
    loss = criterion(decoded_img, test_img[0])
    loss_dist.append(loss.item())
    
    test_img = test_img.cpu().detach().numpy()
    decoded_img = decoded_img.cpu().detach().numpy()
    test_img = test_img/2 + 0.5 # [-1,1]にしているので0.5を足して元に戻す
    decoded_img = decoded_img/2 + 0.5

    test_imgs.append(test_img)
    decoded_imgs.append(decoded_img)

# %%
# lossを視覚化することでどこに異常が隠れているかの情報を得る
loss_sc = []
for i in loss_dist:
    loss_sc.append((i, i))

lower_threshold = 0.0
upper_threshold = 0.09

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
plt.savefig('{}/Threshold.png'.format(save_dir))
fig.show()

'''プロット図
赤線を超えた（右にある）データは異常と判断できる（推定閾値）
青線は下限の閾値
'''

# %%
# 並べて表示したい画像数
num_sumples = 6

plt.figure(figsize=(12, 6))
for i in range(num_sumples):
    # テスト画像を表示
    ax = plt.subplot(3, num_sumples, i + 1)
    plt.imshow(test_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 出力画像を表示
    ax = plt.subplot(3, num_sumples, i + 1 + num_sumples)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 入出力の差分画像を計算
    diff_img = np.abs(test_imgs[i] - decoded_imgs[i])

    # 入出力の差分数値を計算
    diff = np.sum(diff_img)

    # 差分画像と差分数値の表示
    ax = plt.subplot(3, num_sumples, i + 1 + num_sumples * 2)
    plt.imshow(diff_img.reshape(28, 28),cmap="jet")
    #plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_xlabel('score = ' + str(diff))

plt.savefig('{}/result.png'.format(save_dir))
plt.show()
