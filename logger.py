import os

import numpy as np
import cv2
import torch 


class Logger():

    def __init__(self, base_directory):

        self.base_directory = base_directory
        # Create directory to save data
        self.info_directory = os.path.join(self.base_directory, 'info')
        self.color_images_directory = os.path.join(self.base_directory, 'data', 'color-images')
        self.depth_images_directory = os.path.join(self.base_directory, 'data', 'depth-images')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.target_mask_heightmaps_directory = os.path.join(self.base_directory, 'data', 'target-mask-heightmaps')
        self.augment_mask_heightmaps_directory = os.path.join(self.base_directory, 'data', 'augment-mask-heightmaps')
        self.critic_directory = os.path.join(self.base_directory, 'critic_models')
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')
        self.recordings_directory = os.path.join(self.base_directory, 'recordings')
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')
        self.lwrf_results_directory = os.path.join(self.base_directory, 'lwrf_results')
        self.coordinator_directory = os.path.join(self.base_directory, 'coordinator_models')

        if not os.path.exists(self.info_directory):
            os.makedirs(self.info_directory)
        if not os.path.exists(self.color_images_directory):
            os.makedirs(self.color_images_directory)
        if not os.path.exists(self.depth_images_directory):
            os.makedirs(self.depth_images_directory)
        if not os.path.exists(self.color_heightmaps_directory):
            os.makedirs(self.color_heightmaps_directory)
        if not os.path.exists(self.depth_heightmaps_directory):
            os.makedirs(self.depth_heightmaps_directory)
        if not os.path.exists(self.target_mask_heightmaps_directory):
            os.makedirs(self.target_mask_heightmaps_directory)
        if not os.path.exists(self.augment_mask_heightmaps_directory):
            os.makedirs(self.augment_mask_heightmaps_directory)
        if not os.path.exists(self.critic_directory):
            os.makedirs(self.critic_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.recordings_directory):
            os.makedirs(self.recordings_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(os.path.join(self.transitions_directory))
        if not os.path.exists(self.lwrf_results_directory):
            os.makedirs(self.lwrf_results_directory)
        if not os.path.exists(self.coordinator_directory):
            os.makedirs(self.coordinator_directory)

    def save_camera_info(self, intrinsics, pose, depth_scale):
        np.savetxt(os.path.join(self.info_directory, 'camera-intrinsics.txt'), intrinsics, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-pose.txt'), pose, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'camera-depth-scale.txt'), [depth_scale], delimiter=' ')

    def save_heightmap_info(self, boundaries, resolution):
        np.savetxt(os.path.join(self.info_directory, 'heightmap-boundaries.txt'), boundaries, delimiter=' ')
        np.savetxt(os.path.join(self.info_directory, 'heightmap-resolution.txt'), [resolution], delimiter=' ')

    def save_images(self, iteration, color_image, depth_image):
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_images_directory, '%06d.color.png' % iteration), color_image)
        depth_image = np.round(depth_image * 10000).astype(np.uint16) # Save depth in 1e-4 meters
        cv2.imwrite(os.path.join(self.depth_images_directory, '%06d.depth.png' % iteration), depth_image)

    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, target_mask_heightmap):
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, '%06d.color.png' % iteration), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, '%06d.depth.png' % iteration), depth_heightmap)
        cv2.imwrite(os.path.join(self.target_mask_heightmaps_directory, '%06d.mask.png' % iteration), (target_mask_heightmap * 255).astype(np.uint8))

    def save_augment_masks(self, iteration, augment_mask_heightmap):
        cv2.imwrite(os.path.join(self.augment_mask_heightmaps_directory, '%06d.%s.mask.png' % (iteration, 'augment')), (augment_mask_heightmap * 255).astype(np.uint8))

    def write_to_log(self, log_name, log):
        np.savetxt(os.path.join(self.transitions_directory, '%s.log.txt' % log_name), log, delimiter=' ')

    def save_model(self, iteration, model):
        torch.save(model.cpu().state_dict(), os.path.join(self.critic_directory, 'critic-%06d.pth' % iteration))

    def save_visualizations(self, iteration, affordance_vis, name):
        cv2.imwrite(os.path.join(self.visualizations_directory, '%06d.%s.png' % (iteration,name)), affordance_vis)
