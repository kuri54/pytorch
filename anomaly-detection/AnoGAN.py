'''Anomaly Detection of AnoGAN
AnoGAN https://arxiv.org/abs/1703.05921
・正常画像を用いて、乱数ｚから正常な画像を生成できるGeneratorを作成
・作成したGeneratorを用いて、ある画像を入力として与えたときに生成された画像と入力画像との誤差（異常度）を算出
・正常な画像：異常度が小さい
  異常な画像：異常度が大きい
  ことを利用して異常検知をする
  
異常検知のステップ
1. 1次元ベクトルzから正常画像xを生成する方法を学習
2. 画像xをベクトルzにマッピングするために、xに一番近い生成画像G(z)となるzを勾配法で見つける
3. 2.で見つけたzから画像を生成し、入力画像との誤差を異常度とみなして正常／異常を判定
  
問題点
・通常のGANと同様にランダムノイズzから画像を生成する
  -> ある画像を元にその画像に近い画像を生成しようにも、そのある画像を直接入力することができない
     （画像から潜在空間への逆転写ができない）

  AnoGANでは勾配法を用いて、ある画像からその画像に近くなるようなzを探索する
  ・ランダムノイズzを学習済みのGeneratorへ入力し、生成画像を得る
  ・生成画像と入力画像との誤差を計算し、その誤差が小さくなるように勾配降下してzを更新する（必要なだけ繰り返す）
    -> 学習した画像は入力画像に近い画像を生成
       学習していない画像はうまく生成できない
'''
# Fruits-360 dataset をダウンロードし、解凍する
# !wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/rp73yg93n8-1.zip -nc -P ../gan_sample/data
# !unzip -n ./data/rp73yg93n8-1.zip -d ../gan_sample/data
# !unzip -n -q ./data/fruits-360_dataset.zip -d ../gan_sample/data 

# %%
import os
import random
from glob import glob
from warnings import filterwarnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.init as init
from natsort import natsorted
from PIL import Image
from skimage import io, transform
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from torchsummary import summary

filterwarnings('ignore')  # warningをオフにする

# %% 
# ハイパラ
IMAGE_SIZE = 96  # 画像の読み込みサイズ
EMBED_SIZE = 128  # 潜在変数zの次元数
BATCH_SIZE = 16  # バッチサイズ
EPOCHS = 1000  # エポック数
LR = 0.0004  # 学習率


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}  

ATTACH_PATH = './anomaly-detection' # 親ディレクトリ
SAVE_MODEL_PATH = f'{ATTACH_PATH}/results/AnoGAN/model/'  # モデルの保存先
SAVE_IMAGE_FROM_Z_PATH = f'{ATTACH_PATH}/results/AnoGAN/image/image_from_z/'  # 乱数から生成した画像の保存先

# 保存先のディレクトリを作成する
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)  
os.makedirs(SAVE_IMAGE_FROM_Z_PATH, exist_ok=True)  

# %%
train_root = './anomaly-detection/data/fruits-360/Training/Physalis/'  # train dataを保存しているディレクトリ
val_root = './anomaly-detection/data/fruits-360/Test/Physalis/'  # val dataを保存しているディレクトリ

# %%
# ディレクトリから画像を読み込んでDataLoaderに渡す用のクラス
class LoadFromFolder(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = natsorted(os.listdir(main_dir)) # 自然順にソート
        self.all_imgs_name = natsorted(all_imgs)
        self.imgs_loc = [os.path.join(self.main_dir, i) for i in self.all_imgs_name]

    def __len__(self):
        return len(self.all_imgs_name)
    
    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        tensor_image = self.transform(image)
        
        return tensor_image
    
    def __getitem__(self, idx):
        # 後ほどsliceで画像を複数枚取得したいのでsliceでも取れるようにする
        if type(idx) == slice:
            paths = self.imgs_loc[idx]
            tensor_image = [self.load_image(path) for path in paths]
            tensor_image = torch.cat(tensor_image).reshape(len(tensor_image), *tensor_image[0].shape)
            
        elif type(idx) == int:
            path = self.imgs_loc[idx]
            tensor_image = self.load_image(path)
            
        return tensor_image
    
# %%
# 前処理 trainのみHorizontalFlipで水増し
transform_dict = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        ]),
    }

# %%
# dataloaderの作成
train_dataset = LoadFromFolder(train_root, transform=transform_dict['train'])
test_dataset = LoadFromFolder(val_root, transform=transform_dict['test'])

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size = BATCH_SIZE, 
    shuffle=True, 
    **kwargs
    )

val_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size = BATCH_SIZE, 
    shuffle=True, 
    **kwargs
    )

# %%
# Modelの定義  DCGANと同様に乱数zを入力として画像を生成
class Generator(nn.Module):
    def __init__(self, EMBED_SIZE=EMBED_SIZE):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(EMBED_SIZE, 256, kernel_size=6, stride=1, padding=0, bias=False), # 6x6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # 12x12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), # 24x24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), # 48x48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False), #96x96
            nn.Tanh()
        )

    def forward(self, z):
        out = self.main(z)
        return out

# %%
# ネットワークを可視化
summary(Generator().to(device), tuple([EMBED_SIZE, 1, 1]))

# %%
# 入力された画像が正常か異常かの判定結果を返す
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True), #48x48
            nn.Dropout2d(p=0.3),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True), #24x24
            nn.Dropout2d(p=0.3),
            

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True), #12x12
            nn.Dropout2d(p=0.3),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True), #6x6
            nn.Dropout2d(p=0.3),

        )
        self.last = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=6, stride=1, padding=0, bias=False) # 1x1
        )

    def forward(self, x):
        feature = self.main(x)
        out = self.last(feature)
        out = F.sigmoid(out)
        feature = feature.view(feature.size()[0], -1) # Discrimination Lossを算出するため、最終層の1つ前の結果も返す
        out = out.squeeze()
        return out, feature
    
# %%
# ネットワークを可視化
summary(Discriminator().to(device), (3, IMAGE_SIZE, IMAGE_SIZE))

# %%
# 重みの初期化を行う関数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# %%
# モデルの定義
model_G = Generator().to(device)
model_G.apply(weights_init)

model_D = Discriminator().to(device)
model_D.apply(weights_init)

criterion = nn.BCELoss()  # 評価関数
optimizer_g = torch.optim.Adam(model_G.parameters(), lr= LR,betas=(0.5,0.999))  # Generatorのoptimizer
optimizer_d = torch.optim.Adam(model_D.parameters(), lr= LR,betas=(0.5,0.999))  # Discriminatorのoptimizer

# %%
# 学習ループ
loss_d_list, loss_g_list = [], []

for epoch in range(EPOCHS):
    loss_d_sum = 0
    loss_g_sum = 0
    
    for i, (x, x_val) in enumerate(zip(train_loader, val_loader)):
        
        model_G.train()
        model_D.train()
                
        # set values
        y_true = torch.ones(x.size()[0]).to(device)
        y_fake = torch.zeros(x.size()[0]).to(device)
        
        x = x.to(device)
        z = init.normal(torch.Tensor(x.size()[0],EMBED_SIZE, 1, 1),mean=0,std=0.1).to(device)

        # discriminator
        optimizer_d.zero_grad()
        
        G_z = model_G(z)
        p_true, _ = model_D(x)
        p_fake, _ = model_D(G_z)
        
        loss_d = criterion(p_true, y_true) + criterion(p_fake, y_fake)
        loss_d.backward(retain_graph=True)
        optimizer_d.step()
        
        # generator and encoder
        optimizer_g.zero_grad()
        
        p_true, _ = model_D(x)
        p_fake, _ = model_D(G_z)        
        
        loss_g = criterion(p_fake, y_true) + criterion(p_true, y_fake)
        loss_g.backward(retain_graph=True)
        optimizer_g.step()
        
        
        loss_d_sum += loss_d.item()
        loss_g_sum += loss_g.item()
        
            
        # save images
        if i == 0:
            
            model_G.eval()
            model_D.eval()
        
            save_image_size_for_z = min(BATCH_SIZE, 8)
            save_images = model_G(z)
            save_image(save_images[:save_image_size_for_z], f'{SAVE_IMAGE_FROM_Z_PATH}/epoch_{epoch}.png', nrow=4)

        
        
    # record loss
    loss_d_mean = loss_d_sum / len(train_loader)
    loss_g_mean = loss_g_sum / len(train_loader)
    
    print(f'{epoch}/{EPOCHS} epoch g_loss: {loss_g_mean:.3f} d_loss: {loss_d_mean:.3f}')
    
    loss_d_list.append(loss_d_mean)
    loss_g_list.append(loss_g_mean)
    
    # save model
    if (epoch + 1) % 10 == 0:
        torch.save(model_G.state_dict(),f'{SAVE_MODEL_PATH}/Generator_{epoch + 1}.pkl')
        torch.save(model_D.state_dict(),f'{SAVE_MODEL_PATH}/Discriminator_{epoch + 1}.pkl')

# %%
# GeneratorとDiscriminatorのLossの推移
plt.figure(figsize=(12, 8))

plt.plot(range(len(loss_g_list)), loss_g_list, label='g loss')
plt.plot(range(len(loss_d_list)), loss_d_list, label='d loss')
plt.legend()
plt.show()

# %%
# 異常度を測定する関数
criterion_L1 = nn.L1Loss(reduction='sum')

def Anomaly_score(x,G_z,Lambda=0.1):
    _,x_feature = model_D(x)
    _,G_z_feature = model_D(G_z)
    
    residual_loss = criterion_L1(x, G_z)  
    discrimination_loss = criterion_L1(x_feature, G_z_feature)
    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    
    return total_loss

# %%
# 学習したモデルの読み込み
LOAD_EPOCH = 1000

model_G = Generator().to(device)
model_G.load_state_dict(torch.load(f'{SAVE_MODEL_PATH}/Generator_{LOAD_EPOCH}.pkl'))
model_G.eval()


model_D = Discriminator().to(device)
model_D.load_state_dict(torch.load(f'{SAVE_MODEL_PATH}/Discriminator_{LOAD_EPOCH}.pkl'))
model_D.eval()

print('load model')

# %%
# 最適なzを探索する関数
def optimize_z(x):
    x = Variable(x).to(device)
    z = Variable(init.normal(torch.zeros(1,EMBED_SIZE, 1, 1),mean=0,std=0.1).to(device),requires_grad=True)
    z_optimizer = torch.optim.Adam([z],lr=1e-4)
    
    for i in range(1000):
        G_z = model_G(z)

        loss = Anomaly_score(x, G_z)
        loss.backward()
        z_optimizer.step()
            
    return z

# %%
# 正常な画像で実行
random_image_size = 10

test_root_normal = './anomaly-detection/data/fruits-360/Test/Physalis/'
test_dataset_normal = LoadFromFolder(test_root_normal, transform=transform_dict['test'])

test_images_normal = random.sample(list(test_dataset_normal), random_image_size)

# うまく再現され、異常スコアが低くなっていれば成功
for idx in range(len(test_images_normal)):

    x = test_images_normal[idx].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    x = Variable(x).to(device)
    
    z_o = optimize_z(x)
    G_z_o = model_G(z_o)
    loss = Anomaly_score(x, G_z_o)
    diff_img = torch.abs(x - G_z_o)

    print(f'Anomary_score = {loss.cpu().data:.3f}')
    comparison = torch.cat([x.to('cpu'), G_z_o.to('cpu'), diff_img.to('cpu')])
    joined_image = make_grid(comparison, nrow=3).detach().numpy()
    joined_image = np.transpose(joined_image, [1, 2, 0])
    
    plt.figure(figsize=(12, 4))
    plt.imshow((joined_image * 255).astype(np.uint8))
    plt.show()

# %%
# 異常な画像で実行
random_image_size = 10

test_root_anomaly = './anomaly-detection/data/fruits-360/Test/Apple Braeburn/'
test_dataset_anomaly = LoadFromFolder(test_root_anomaly, transform=transform_dict['test'])

test_images_anomaly = random.sample(list(test_dataset_anomaly), random_image_size)


# うまく再現されず、異常スコアが高くなっていれば成功
for idx in range(len(test_images_anomaly)):

    x = test_images_anomaly[idx].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    x = Variable(x).to(device)
    
    z_o = optimize_z(x)
    G_z_o = model_G(z_o)
    loss = Anomaly_score(x, G_z_o)
    diff_img = torch.abs(x - G_z_o)

    print(f'Anomary_score = {loss.cpu().data:.3f}')
    comparison = torch.cat([x.to('cpu'), G_z_o.to('cpu'), diff_img.to('cpu')])
    joined_image = make_grid(comparison, nrow=3).detach().numpy()
    joined_image = np.transpose(joined_image, [1, 2, 0])
    
    plt.figure(figsize=(12, 4))
    plt.imshow((joined_image * 255).astype(np.uint8))
    plt.show()

# %%
# 画像に傷を模した記号を付与する関数
def add_damage(image_path):
    
    folder = os.path.dirname(image_path)
    save_folder = folder + '_damaged'
    save_base_path = os.path.basename(image_path).replace('.jpg', '_damaged.jpg')
    save_path = os.path.join(save_folder, save_base_path)
    
    os.makedirs(save_folder, exist_ok=True)
    
    image = cv2.imread(image_path)
    center_x = random.randint(20, 76)
    center_y = random.randint(20, 76)
    color_r = random.randint(0, 255)
    color_g = random.randint(0, 255)
    color_b = random.randint(0, 255)
    
    center = (center_x, center_y)
    color = (color_r, color_g, color_b)
    
    cv2.circle(image, center = center, radius = 10, color = color,thickness=-1)
    cv2.imwrite(save_path, image)
    
images_path = glob('./anomaly-detection/data/fruits-360/Test/Physalis/*.jpg')
[add_damage(image_path) for image_path in images_path]
print('add damage')

# %%
# 異常な画像で実行
test_root_anomaly = './anomaly-detection/data/fruits-360/Test/Physalis_damaged/'
test_dataset_anomaly = LoadFromFolder(test_root_anomaly, transform=transform_dict['test'])

test_images_anomaly = random.sample(list(test_dataset_anomaly), random_image_size)

# うまく再現されず、異常スコアが高くなっていれば成功
for idx in range(len(test_images_anomaly)):

    x = test_images_anomaly[idx].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    x = Variable(x).to(device)
    
    z_o = optimize_z(x)
    G_z_o = model_G(z_o)
    loss = Anomaly_score(x, G_z_o)
    diff_img = torch.abs(x - G_z_o)

    print(f'Anomary_score = {loss.cpu().data:.3f}')
    comparison = torch.cat([x.to('cpu'), G_z_o.to('cpu'), diff_img.to('cpu')])
    joined_image = make_grid(comparison, nrow=3).detach().numpy()
    joined_image = np.transpose(joined_image, [1, 2, 0])
    
    plt.figure(figsize=(12, 4))
    plt.imshow((joined_image * 255).astype(np.uint8))
    plt.show()
