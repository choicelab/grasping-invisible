import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from light_weight_refinenet.models.resnet import rf_lw50
from light_weight_refinenet.utils.visualize import *
from light_weight_refinenet.utils.helpers import prepare_img


class LwrfInfer:
    def __init__(self, use_cuda, save_path=None):
        # class_ids: 1 to num_classes, 0 is background
        self.num_classes = 7
        self.net = nn.DataParallel(rf_lw50(num_classes=10))

        # hard-coded paths
        weights_path = './light_weight_refinenet/resnet50.pth.tar'
        self.color_map = np.load('./light_weight_refinenet/utils/color_map.npy')
        self.class_map = np.load('./light_weight_refinenet/utils/class_map.npy', allow_pickle=True).item()

        if use_cuda:
            self.net = self.net.cuda()
            self.net.load_state_dict(torch.load(weights_path)['segmenter'])
        else:
            self.net.load_state_dict(torch.load(weights_path, map_location='cpu')['segmenter'])

        self.net.eval()

        self.count = 0
        self.fig, self.ax = plt.subplots(1, figsize=(16, 16))
        if not save_path:
            self.save_path = './lwrf_results/'
        else:
            self.save_path = save_path

    def segment(self, img):
        detected_thold = 50
        self.image = img
        orig_size = self.image.shape[:2][::-1]
        img_inp = torch.from_numpy(prepare_img(self.image).transpose(2, 0, 1)[None]).float()

        segm = self.net(img_inp)[0].cpu().detach().numpy().transpose(1, 2, 0)
        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
        segm = segm.argmax(axis=2).astype(np.uint8)

        class_ids = []
        labels = []
        masks = np.zeros((segm.shape[0], segm.shape[1], self.num_classes))
        count = 0
        for class_id in range(1, self.num_classes+1):
            temp = np.zeros((segm.shape[0], segm.shape[1]))
            temp[segm == class_id] = 1
            if np.sum(temp) > detected_thold:
                masks[:, :, count] = temp
                class_ids.append(class_id)
                labels.append(self.class_map[class_id])
                count += 1

        self.res = {'class_ids': class_ids, 'labels': labels, 'masks': masks}

        return self.res

    def display_instances(self, overlay=False, title=""):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        colors = self.color_map[np.array(self.res['class_ids'])]/255
        if overlay:
            masked_image = self.image.astype(np.uint32).copy()
        else:
            masked_image = np.zeros_like(self.image).astype(np.uint32)

        for i in range(len(self.res['labels'])):
            color = colors[i]

            mask = self.res['masks'][:, :, i]
            masked_image = apply_mask(masked_image, mask, color)

        self.ax.imshow(masked_image.astype(np.uint8))
        self.fig.savefig(os.path.join(self.save_path, title + '.png'))
        self.ax.clear()


