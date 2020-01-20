import torch
from models.resnet import rf_lw50, rf_lw152
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.helpers import prepare_img
import cv2
import torch.nn as nn
import os

net = nn.DataParallel(rf_lw152(num_classes=10))
if torch.cuda.is_available:
    net = net.cuda()
    net.load_state_dict(torch.load('./ckpt/resnet152.tar')['segmenter'])
else:
    net.load_state_dict(torch.load('./ckpt/resnet152.tar', map_location='cpu')['segmenter'])
net.eval()

color_map = np.load('../utils/color_map.npy')

img_dir = '../examples/imgs/VrepYCB/augment/'
imgs = []
for img in os.listdir(img_dir):
    if img.endswith('.jpg'):
        imgs.append(os.path.join(img_dir, img))

n_rows = len(imgs)

plt.figure(figsize=(16, 12))
idx = 1

with torch.no_grad():
    for img_path in sorted(imgs):
        img = np.array(Image.open(img_path))
        msk = color_map[np.array(Image.open(img_path.replace('.jpg', '.png')))]
        orig_size = img.shape[:2][::-1]

        img_inp = torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]).float()

        plt.subplot(n_rows, 3, idx)
        plt.imshow(img)
        plt.title('img')
        plt.axis('off')
        idx += 1

        plt.subplot(n_rows, 3, idx)
        plt.imshow(msk)
        plt.title('gt')
        plt.axis('off')
        idx += 1

        segm = net(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
        segm = color_map[segm.argmax(axis=2).astype(np.uint8)]

        plt.subplot(n_rows, 3, idx)
        plt.imshow(segm)
        plt.title('prediction')
        plt.axis('off')
        idx += 1
plt.show()
