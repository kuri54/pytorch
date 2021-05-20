import sys

sys.path
sys.path.append('/work/pytorch/grad-cam')

from torchvision.models import *
from visualisation.core.utils import device
from efficientnet_pytorch import EfficientNet
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch 
from utils import *
import PIL.Image
import cv2

from visualisation.core.utils import device 
from visualisation.core.utils import image_net_postprocessing

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from visualisation.core import *
from visualisation.core.utils import image_net_preprocessing

import os

# for animation
%matplotlib inline
from IPython.display import Image
from matplotlib.animation import FuncAnimation
from collections import OrderedDict

def efficientnet(model_name='efficientnet-b0',**kwargs):
    return EfficientNet.from_pretrained(model_name).to(device)

max_img = 5
path = '../../datasets/stanford-dogs/'
interesting_categories = os.listdir(os.path.join(path, 'test'))
interesting_categories

images = [] 
for category_name in interesting_categories:
    image_paths = glob.glob(f'{path}/test/{category_name}/*')
    category_images = list(map(lambda x: PIL.Image.open(x), image_paths[:max_img]))
    images.extend(category_images)

inputs  = [Compose([Resize((224,224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]

model_outs = OrderedDict()
model_instances = [alexnet, densenet121, resnet152, 
                   lambda pretrained:efficientnet(model_name='efficientnet-b0'),
                   lambda pretrained:efficientnet(model_name='efficientnet-b3')]
                   
model_names = [m.__name__ for m in model_instances]
model_names[-2],model_names[-1] = 'EB0','EB3'

images = list(map(lambda x: cv2.resize(np.array(x),(224,224)),images)) # resize i/p img

for name,model in zip(model_names,model_instances):
    
    module = model(pretrained=True).to(device)
    module.eval()

    vis = GradCam(module, device)

    model_outs[name] = list(map(lambda x: tensor2img(vis(x, None,postprocessing=image_net_postprocessing)[0]), inputs))
    del module
    torch.cuda.empty_cache()


# create a figure with two subplots
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6,figsize=(20,20))
axes = [ax2, ax3, ax4, ax5, ax6]

def update(frame):
    all_ax = []
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.text(1, 1, 'Orig. Im', color="white", ha="left", va="top",fontsize=30)
    all_ax.append(ax1.imshow(images[frame]))
    for i,(ax,name) in enumerate(zip(axes,model_outs.keys())):
        ax.set_yticklabels([])
        ax.set_xticklabels([])        
        ax.text(1, 1, name, color="white", ha="left", va="top",fontsize=20)
        all_ax.append(ax.imshow(model_outs[name][frame], animated=True))

    return all_ax

ani = FuncAnimation(fig, update, frames=range(len(images)), interval=1000, blit=True)
model_names = [m.__name__ for m in model_instances]
model_names = ', '.join(model_names)
fig.tight_layout()
ani.save('compare_arch.gif', writer='imagemagick')

Image('compare_arch.gif')
