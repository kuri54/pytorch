# %%
import os
import pylab
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset

from torchvision import transforms
from torchvision.datasets import MNIST

# %%
class Mnisttox(Dataset):
    def __init__(self, datasets ,labels:list):
        self.dataset = [datasets[i][0] for i in range(len(datasets))
                        if datasets[i][1] in labels ]
        self.labels = labels
        self.len_oneclass = int(len(self.dataset)/10)

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        img = self.dataset[index]
        return img,[]

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
mount_dir = './anomaly-detection'

batch_size = 256
num_epochs = 100
learning_rate = 0.001
num_sumples = 6 #number of test sample

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
    transforms.Normalize((0.5, ), (0.5, ))  # [0,1] => [-1,1]
])
train_dataset = MNIST('./anomaly-detection/data', download=True, train=True, transform=img_transform)
train_1 = Mnisttox(train_dataset,[1])
train_loader = DataLoader(train_1, batch_size=batch_size, shuffle=True)

# %%
model.train()
losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for img, _ in train_loader:
        # print("now")
        img = img.to(device)
        
        optimizer.zero_grad()
        
        input_img = Variable(img.view(img.size(0), -1))
        decoded_img = model(input_img)

        # 出力画像（再構成画像）と入力画像の間でlossを計算
        loss = criterion(decoded_img, input_img)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)

    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        num_epochs,
        epoch_loss))

# %%
fig, ax = plt.subplots(1,1,figsize=(7,5))
ax.set_title('Loss')
ax.plot(losses)
plt.savefig('./anomaly-detection/save/loss.pdf')

# %%
test_dataset = MNIST('./anomaly-detection/data', train=False, download=True, transform=img_transform)
test_1_9 = Mnisttox(test_dataset,[1,9])
test_loader = DataLoader(test_1_9, batch_size=len(test_dataset), shuffle=True)

# %%
model.eval()
loss_dist = []
for img, _ in test_loader:
    img = img.to(device)
    test_img = Variable(img.view(img.size(0), -1)).to(device)

    decoded_img = model(test_img)
        
    test_img = test_img.cpu().detach().numpy()
    decoded_img = decoded_img.cpu().detach().numpy()
    test_img = test_img/2 + 0.5
    decoded_img = decoded_img/2 + 0.5
    
# %%
plt.figure(figsize=(12, 6))
for i in range(num_sumples):
    # テスト画像を表示
    ax = plt.subplot(3, num_sumples, i + 1)
    plt.imshow(test_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 出力画像を表示
    ax = plt.subplot(3, num_sumples, i + 1 + num_sumples)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 入出力の差分画像を計算
    diff_img = np.abs(test_img[i] - decoded_img[i])

    # 入出力の差分数値を計算
    diff = np.sum(diff_img)

    # 差分画像と差分数値の表示
    ax = plt.subplot(3, num_sumples, i + 1 + num_sumples * 2)
    plt.imshow(diff_img.reshape(28, 28),cmap="jet")
    #plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.set_xlabel('score = ' + str(diff))

plt.savefig(mount_dir+"/save/result.png")
plt.show()
