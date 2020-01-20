import math

import numpy as np
from skimage.morphology.convex_hull import convex_hull_image
from scipy.ndimage.morphology import binary_dilation


def check_grasp_margin(target_mask_heightmap, depth_heightmap):
    margin_mask = binary_dilation(target_mask_heightmap, iterations=10).astype(np.float32)-target_mask_heightmap
    margin_depth = margin_mask * depth_heightmap
    margin_depth[np.isnan(margin_depth)] = 0
    margin_depth[margin_depth > 0.3] = 0
    margin_depth[margin_depth < 0.02] = 0
    margin_depth[margin_depth > 0] = 1
    margin_value = np.sum(margin_depth)
    return margin_value/np.sum(margin_mask), margin_value/np.sum(target_mask_heightmap)


def check_push_target_oriented(best_pix_ind, push_end_pix_yx, target_mask_heightmap, mask_count_threshold=5):
    mask_hull = convex_hull_image(target_mask_heightmap)
    mask_count = 0
    x1 = best_pix_ind[2]
    y1 = best_pix_ind[1]
    x2 = push_end_pix_yx[1]
    y2 = push_end_pix_yx[0]
    x_range = abs(x2-x1)
    y_range = abs(y2-y1)
    if x_range > y_range:
        k = (y2-y1)/(x2-x1)
        b = y1-k*x1
        for x in range(min(int(x1), int(x2)), max(int(x1), int(x2))+1):
            y = int(k*x+b)
            try:
                mask_count += mask_hull[y, x]
            except IndexError:
                pass
    else:
        k = (x2-x1)/(y2-y1)
        b = x1-k*y1
        for y in range(min(int(y1), int(y2)), max(int(y1), int(y2))+1):
            x = int(k*y+b)
            try:
                mask_count += mask_hull[y, x]
            except IndexError:
                pass
    if mask_count > mask_count_threshold:
        return True
    else:
        return False


def check_grasp_target_oriented(best_pix_ind, target_mask_heightmap):
    mask_hull = convex_hull_image(target_mask_heightmap)
    if mask_hull[int(best_pix_ind[1]), int(best_pix_ind[2])]:
        return True
    else:
        return False


def get_push_pix(push_maps, num_rotations):
    push_pix_ind = np.unravel_index(np.argmax(push_maps), push_maps.shape)
    push_end_pix_yx = get_push_end_pix_yx(push_pix_ind, num_rotations)
    return push_pix_ind, push_end_pix_yx


def get_push_end_pix_yx(push_pix_ind, num_rotations):
    push_orientation = [1.0, 0.0]
    push_length_pix = 0.1/0.002
    rotation_angle = np.deg2rad(push_pix_ind[0]*(360.0/num_rotations))
    push_direction = np.asarray([push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
                                 push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle)])
    return [push_pix_ind[1] + push_direction[1] * push_length_pix, push_pix_ind[2] + push_direction[0] * push_length_pix]


def check_env_depth_change(prev_depth_heightmap, depth_heightmap, change_threshold=300):
    depth_diff = abs(prev_depth_heightmap-depth_heightmap)
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.3] = 0
    depth_diff[depth_diff < 0.02] = 0
    depth_diff[depth_diff > 0] = 1
    change_value = np.sum(depth_diff)
    change_detected = change_value > change_threshold

    return change_detected, change_value


def check_target_depth_change(prev_depth_heightmap, prev_target_mask_heightmap, depth_heightmap, change_threshold=50):
    prev_mask_hull = binary_dilation(convex_hull_image(prev_target_mask_heightmap), iterations=5)
    depth_diff = prev_mask_hull*(prev_depth_heightmap-depth_heightmap)
    depth_diff[np.isnan(depth_diff)] = 0
    depth_diff[depth_diff > 0.3] = 0
    depth_diff[depth_diff < 0.02] = 0
    depth_diff[depth_diff > 0] = 1
    change_value = np.sum(depth_diff)
    change_detected = change_value > change_threshold

    return change_detected, change_value


def process_mask_heightmaps(segment_results, seg_mask_heightmaps):
    names = []
    heightmaps = []
    for i in range(len(segment_results['labels'])):
        name = segment_results['labels'][i]
        heightmap = seg_mask_heightmaps[:, :, i]
        if np.sum(heightmap) > 10:
            names.append(name)
            heightmaps.append(heightmap)
    return {'names': names, 'heightmaps': heightmaps}


def get_replay_id(predicted_value_log, label_value_log, reward_value_log, sample_ind, replay_type):
    # Prioritized experience replay, find sample with highest surprise value
    sample_ind = np.asarray(sample_ind)
    predicted_values = np.asarray(predicted_value_log)[sample_ind]
    label_values = np.asarray(label_value_log)[sample_ind]
    reward_values = np.asarray(reward_value_log)[sample_ind]
    if replay_type == 'augment':
        # assume predicted_value for different mask input are close
        label_values = label_values - reward_values + 1.0

    sample_surprise_values = np.abs(predicted_values - label_values)
    sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
    sorted_sample_ind = sample_ind[sorted_surprise_ind]
    pow_law_exp = 2
    rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1) * (sample_ind.size - 1)))
    sample_iteration = sorted_sample_ind[rand_sample_ind]
    print(replay_type.capitalize(), 'replay: iteration %d (surprise value: %f)' %
          (sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
    return sample_iteration


def get_pointcloud(color_img, depth_img, masks_imgs, camera_intrinsics):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w-1, im_w), np.linspace(0, im_h-1, im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w, 1)
    cam_pts_y.shape = (im_h*im_w, 1)
    cam_pts_z.shape = (im_h*im_w, 1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0]
    rgb_pts_g = color_img[:, :, 1]
    rgb_pts_b = color_img[:, :, 2]
    rgb_pts_r.shape = (im_h*im_w, 1)
    rgb_pts_g.shape = (im_h*im_w, 1)
    rgb_pts_b.shape = (im_h*im_w, 1)

    num_masks = masks_imgs.shape[2]
    masks_pts = masks_imgs.copy()
    masks_pts = masks_pts.transpose(2, 0, 1).reshape(num_masks, -1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts, masks_pts


def get_heightmap(color_img, depth_img, masks_imgs, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):

    num_masks = masks_imgs.shape[2]

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts, masks_pts = get_pointcloud(color_img, depth_img, masks_imgs, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    masks_pts = masks_pts[:, sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    masks_pts = masks_pts[:, heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    masks_heightmaps = np.zeros((heightmap_size[0], heightmap_size[1], num_masks), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:, [0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:, [1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:, [2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    for c in range(num_masks):
        masks_heightmaps[heightmap_pix_y, heightmap_pix_x, c] = masks_pts[c, :]
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap, masks_heightmaps


# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
