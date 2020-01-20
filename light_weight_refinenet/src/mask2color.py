import numpy as np
import os
from PIL import Image
import cv2

mask_files_path = ''
out_path = ''
color_map = np.load('../utils/color_map.npy')

for mask_file in os.listdir(mask_files_path):
    if mask_file.endswith('.png'):
        img_path = os.path.join(mask_files_path, mask_file)
        msk = color_map[np.array(Image.open(img_path))]
        cv2.imwrite(os.path.join(out_path, mask_file), msk)
