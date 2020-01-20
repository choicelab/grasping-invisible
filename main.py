#!/usr/bin/env python

import time
import datetime
import os
import random
import threading
import argparse

import torch
import numpy as np
import cv2

from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from lwrf_infer import LwrfInfer
from policies import Explorer, Coordinator


def main(args):

    # --------------- Setup options ---------------
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
    heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    future_reward_discount = args.future_reward_discount
    stage_epoch = args.stage_epoch

    # -------------- Object options --------------
    config_file = args.config_file

    # -------------- Testing options --------------
    is_testing = args.is_testing
    test_preset_cases = args.test_preset_cases
    test_target_seeking = args.test_target_seeking
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    max_motion_onecase = args.max_motion_onecase

    # ------ Pre-loading and logging options ------
    load_ckpt = args.load_ckpt  # Load pre-trained ckpt of model
    critic_ckpt_file = os.path.abspath(args.critic_ckpt) if load_ckpt else None
    coordinator_ckpt_file = os.path.abspath(args.coordinator_ckpt) if load_ckpt else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    save_visualizations = args.save_visualizations

    print('-----------------------')
    if not is_testing:
        if continue_logging:
            logging_directory = os.path.abspath(args.logging_directory)
            print('Pre-loading data logging session: %s' % logging_directory)
        else:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            logging_directory = os.path.join(os.path.abspath('logs'), timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
            print('Creating data logging session: %s' % logging_directory)
    else:
        logging_directory = os.path.join(os.path.abspath('logs'), 'testing/release', config_file.split('/')[-1].split('.')[0])
        print('Creating data logging session: %s' % logging_directory)

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(workspace_limits, is_testing, test_preset_cases, config_file)

    # Initialize trainer
    trainer = Trainer(future_reward_discount, is_testing, load_ckpt, critic_ckpt_file, force_cpu)

    # Initialize data logger
    logger = Logger(logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale)  # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Define light weight refinenet model
    lwrf_model = LwrfInfer(use_cuda=trainer.use_cuda, save_path=logger.lwrf_results_directory)

    # Define exploration policy (search for the invisible target)
    explorer = Explorer(map_size=int(round((workspace_limits[0, 1] - workspace_limits[0, 0]) / heightmap_resolution)))

    # Define coordination policy (coordinate target-oriented pushing and grasping)
    coordinator = Coordinator(save_dir=logger.coordinator_directory, ckpt_file=coordinator_ckpt_file)

    # Initialize variables for grasping fail and exploration probability
    grasp_fail_count = [0]
    motion_fail_count = [0]
    explore_prob = 0.505 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'seeking_target': False,
                          'best_push_pix_ind': None,
                          'push_end_pix_yx': None,
                          'margin_occupy_ratio': None,
                          'margin_occupy_norm': None,
                          'best_grasp_pix_ind': None,
                          'best_pix_ind': None,
                          'target_grasped': False}

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:

                # Get pixel location and rotation with highest affordance prediction
                nonlocal_variables['best_push_pix_ind'], nonlocal_variables['push_end_pix_yx'] = utils.get_push_pix(push_predictions, trainer.model.num_rotations)
                nonlocal_variables['best_grasp_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_push_prediction_vis(push_predictions, color_heightmap, nonlocal_variables['best_push_pix_ind'], nonlocal_variables['push_end_pix_yx'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_grasp_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_grasp_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

                if nonlocal_variables['seeking_target']:
                    print('Seeking target in testing mode')
                    nonlocal_variables['primitive_action'] = 'push'
                    height_priors = trainer.push_heuristic(valid_depth_heightmap)
                    prior = np.multiply(height_priors, push_predictions)
                    post = explorer.get_action_maps(prior)
                    search_push_pix_ind, search_push_end_pix_yx = utils.get_push_pix(post, trainer.model.num_rotations)
                    explorer.update(search_push_end_pix_yx)
                    if save_visualizations:
                        search_push_pred_vis = trainer.get_push_prediction_vis(post, color_heightmap, search_push_pix_ind, search_push_end_pix_yx)
                        cv2.imwrite('visualization.search.png', search_push_pred_vis)

                    nonlocal_variables['best_pix_ind'] = search_push_pix_ind
                else:
                    # Determine whether grasping or pushing should be executed based on network predictions
                    best_push_conf = np.max(push_predictions)
                    best_grasp_conf = np.max(grasp_predictions)
                    print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))

                    # Actor
                    if not is_testing and trainer.iteration < stage_epoch:
                        print('Greedy deterministic policy ...')
                        motion_type = 1 if best_grasp_conf > best_push_conf else 0
                    else:
                        print('Coordination policy ...')
                        syn_input = [best_push_conf, best_grasp_conf, nonlocal_variables['margin_occupy_ratio'], nonlocal_variables['margin_occupy_norm'], grasp_fail_count[0]]
                        motion_type = coordinator.predict(syn_input)
                    explore_actions = np.random.uniform() < explore_prob
                    if explore_actions:
                        print('Exploring actions, explore_prob: %f' % explore_prob)
                        motion_type = 1-0

                    nonlocal_variables['primitive_action'] = 'push' if motion_type == 0 else 'grasp'

                    if nonlocal_variables['primitive_action'] == 'push':
                        grasp_fail_count[0] = 0
                        nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_push_pix_ind']
                        predicted_value = np.max(push_predictions)
                    elif nonlocal_variables['primitive_action'] == 'grasp':
                        nonlocal_variables['best_pix_ind'] = nonlocal_variables['best_grasp_pix_ind']
                        predicted_value = np.max(grasp_predictions)

                    # Save predicted confidence value
                    trainer.predicted_value_log.append([predicted_value])
                    logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'push':
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0]
                    else:
                        safe_z_position = np.max(local_region) + workspace_limits[2][0]
                    primitive_position[2] = safe_z_position

                # Save executed primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Initialize variables that influence reward
                nonlocal_variables['target_grasped'] = False

                motion_fail_count[0] += 1
                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    robot.push(primitive_position, best_rotation_angle, workspace_limits)
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    grasp_fail_count[0] += 1
                    grasped_object_name = robot.grasp(primitive_position, best_rotation_angle, workspace_limits)
                    if grasped_object_name in segment_results['labels']:
                        print('Grasping succeed, grasped', grasped_object_name)
                        nonlocal_variables['target_grasped'] = grasped_object_name == target_name
                        print('Target grasped?:', nonlocal_variables['target_grasped'])
                        if nonlocal_variables['target_grasped']:
                            motion_fail_count[0] = 0
                            grasp_fail_count[0] = 0
                        else:
                            # posthoc labeling for data augmentation
                            augment_id = segment_results['labels'].index(grasped_object_name)
                            augment_mask_heightmap = seg_mask_heightmaps[:, :, augment_id]
                            logger.save_augment_masks(trainer.iteration, augment_mask_heightmap)
                            trainer.augment_ids.append(trainer.iteration)
                            logger.write_to_log('augment-ids', trainer.augment_ids)
                    else:
                        print('Grasping failed')

                trainer.target_grasped_log.append(int(nonlocal_variables['target_grasped']))
                logger.write_to_log('target-grasped', trainer.target_grasped_log)

                # Data for classifier actor
                if not is_testing and trainer.iteration >= stage_epoch:
                    robot.sim_read_config_file(config_file='simulation/random/random-8blocks.txt')
                    if nonlocal_variables['primitive_action'] == 'grasp' and utils.check_grasp_target_oriented(nonlocal_variables['best_pix_ind'], target_mask_heightmap):
                        data_label = int(nonlocal_variables['target_grasped'])
                        print('Collecting classifier data', data_label)

                        coordinator.memory.push(syn_input, data_label)

                nonlocal_variables['executing_action'] = False

            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # Replay training function
    # -------------------------------------------------------------
    def replay_training(replay_id, replay_primitive_action, replay_type=None):
        # Load replay RGB-D and mask heightmap
        replay_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.color.png' % (replay_id)))
        replay_color_heightmap = cv2.cvtColor(replay_color_heightmap, cv2.COLOR_BGR2RGB)
        replay_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.depth.png' % (replay_id)), -1)
        replay_depth_heightmap = replay_depth_heightmap.astype(np.float32) / 100000
        if replay_type == 'augment':
            replay_mask_heightmap = cv2.imread(os.path.join(logger.augment_mask_heightmaps_directory, '%06d.augment.mask.png' % (replay_id)), -1)
        else:
            replay_mask_heightmap = cv2.imread(os.path.join(logger.target_mask_heightmaps_directory, '%06d.mask.png' % (replay_id)), -1)
        replay_mask_heightmap = replay_mask_heightmap.astype(np.float32) / 255

        replay_reward_value = trainer.reward_value_log[replay_id][0]
        if replay_type == 'augment':
            # reward for target_grasped is 1.0
            replay_reward_value = 1.0

        # Read next states
        next_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.color.png' % (replay_id+1)))
        next_color_heightmap = cv2.cvtColor(next_color_heightmap, cv2.COLOR_BGR2RGB)
        next_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.depth.png' % (replay_id+1)), -1)
        next_depth_heightmap = next_depth_heightmap.astype(np.float32) / 100000
        next_mask_heightmap = cv2.imread(os.path.join(logger.target_mask_heightmaps_directory, '%06d.mask.png' % (replay_id+1)), -1)
        next_mask_heightmap = next_mask_heightmap.astype(np.float32) / 255

        replay_change_detected, _ = utils.check_env_depth_change(replay_depth_heightmap, next_depth_heightmap)

        if not replay_change_detected:
            replay_future_reward = 0.0
        else:
            replay_next_push_predictions, replay_next_grasp_predictions, _ = trainer.forward(
                next_color_heightmap, next_depth_heightmap, next_mask_heightmap, is_volatile=True)
            replay_future_reward = max(np.max(replay_next_push_predictions), np.max(replay_next_grasp_predictions))
        new_sample_label_value = replay_reward_value + trainer.future_reward_discount * replay_future_reward

        # Get labels for replay and backpropagate
        replay_best_pix_ind = (np.asarray(trainer.executed_action_log)[replay_id, 1:4]).astype(int)
        trainer.backprop(replay_color_heightmap, replay_depth_heightmap, replay_mask_heightmap,
                         replay_primitive_action, replay_best_pix_ind, new_sample_label_value)

        # Recompute prediction value and label for replay buffer
        # Compute forward pass with replay
        replay_push_predictions, replay_grasp_predictions, _ = trainer.forward(
            replay_color_heightmap, replay_depth_heightmap, replay_mask_heightmap, is_volatile=True)
        if replay_primitive_action == 'push':
            trainer.predicted_value_log[replay_id] = [np.max(replay_push_predictions)]
            trainer.label_value_log[sample_iteration] = [new_sample_label_value]
        elif replay_primitive_action == 'grasp':
            trainer.predicted_value_log[replay_id] = [np.max(replay_grasp_predictions)]
            trainer.label_value_log[sample_iteration] = [new_sample_label_value]

    # Reposition function
    # -------------------------------------------------------------
    def reposition_objects():
        robot.restart_sim()
        robot.add_objects()
        grasp_fail_count[0] = 0
        motion_fail_count[0] = 0
        trainer.reposition_log.append([trainer.iteration])
        logger.write_to_log('reposition', trainer.reposition_log)

    augment_training = False
    target_name = None
    # Start main training/testing loop
    # -------------------------------------------------------------
    while True:

        if test_target_seeking and nonlocal_variables['target_grasped']:
            # Restart if target grasped in test_target_seeking mode
            reposition_objects()
            target_name = None
            explorer.reset()
            if is_testing:
                trainer.model.load_state_dict(torch.load(critic_ckpt_file))
            del prev_color_img

        # program stopping criterion
        if is_testing and len(trainer.reposition_log) >= max_test_trials:
            return

        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

        # Use lwrf to segment/detect target object
        segment_results = lwrf_model.segment(color_img)

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap, seg_mask_heightmaps = utils.get_heightmap(
            color_img, depth_img, segment_results['masks'], robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        mask_heightmaps = utils.process_mask_heightmaps(segment_results, seg_mask_heightmaps)

        # Check targets
        if (len(mask_heightmaps['names']) == 0 and not test_target_seeking) or motion_fail_count[0] >= max_motion_onecase:
            # Restart if no targets detected
            reposition_objects()
            target_name = None
            if is_testing:
                trainer.model.load_state_dict(torch.load(critic_ckpt_file))
            continue

        # Choose target
        if len(mask_heightmaps['names']) == 0 and test_target_seeking:
            nonlocal_variables['seeking_target'] = True
            target_mask_heightmap = np.ones_like(valid_depth_heightmap)
        else:
            nonlocal_variables['seeking_target'] = False

            # lwrf_model.display_instances(title=str(trainer.iteration))

            if target_name in mask_heightmaps['names']:
                target_mask_heightmap = mask_heightmaps['heightmaps'][mask_heightmaps['names'].index(target_name)]
            else:
                target_id = random.randint(0, len(mask_heightmaps['names'])-1)
                target_name = mask_heightmaps['names'][target_id]
                target_mask_heightmap = mask_heightmaps['heightmaps'][target_id]
            print('lwrf segments:', mask_heightmaps['names'])
            print('Target name:', target_name)

            nonlocal_variables['margin_occupy_ratio'], nonlocal_variables['margin_occupy_norm'] = utils.check_grasp_margin(target_mask_heightmap, depth_heightmap)

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img)
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, target_mask_heightmap)

        # Run forward pass with network to get affordances
        push_predictions, grasp_predictions, state_feat = trainer.forward(
            color_heightmap, valid_depth_heightmap, target_mask_heightmap, is_volatile=True)

        # Execute best primitive action on robot in another thread
        nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        # -------------------------------------------------------------
        if 'prev_color_img' in locals():

            motion_target_oriented = False
            if prev_primitive_action == 'push':
                motion_target_oriented = utils.check_push_target_oriented(prev_best_pix_ind, prev_push_end_pix_yx, prev_target_mask_heightmap)
            elif prev_primitive_action == 'grasp':
                motion_target_oriented = utils.check_grasp_target_oriented(prev_best_pix_ind, prev_target_mask_heightmap)

            margin_increased = False
            if not robot.objects_reset:
                # Detect push changes
                if not prev_target_grasped:
                    margin_increase_threshold = 0.1
                    margin_increase_val = prev_margin_occupy_ratio-nonlocal_variables['margin_occupy_ratio']
                    if margin_increase_val > margin_increase_threshold:
                        margin_increased = True
                        print('Grasp margin increased: (value: %d)' % margin_increase_val)

            push_effective = margin_increased

            env_change_detected, _ = utils.check_env_depth_change(prev_depth_heightmap, depth_heightmap)

            # Compute training labels
            label_value, prev_reward_value = trainer.get_label_value(
                prev_primitive_action, motion_target_oriented, env_change_detected, push_effective, prev_target_grasped,
                color_heightmap, valid_depth_heightmap, target_mask_heightmap)
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)

            # Backpropagate
            # regular backprop
            l = trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_target_mask_heightmap,
                                 prev_primitive_action, prev_best_pix_ind, label_value)
            trainer.loss_queue.append(l)
            trainer.loss_rec.append(sum(trainer.loss_queue) / len(trainer.loss_queue))
            logger.write_to_log('loss-rec', trainer.loss_rec)

            # Adjust exploration probability
            if not is_testing:
                explore_prob = 0.5 * np.power(0.998, trainer.iteration) + 0.05

            # Do sampling for experience replay
            if not is_testing:
                sample_primitive_action = prev_primitive_action
                if sample_primitive_action == 'push':
                    sample_primitive_action_id = 0
                elif sample_primitive_action == 'grasp':
                    sample_primitive_action_id = 1

                # Get samples of the same primitive but with different results
                sample_ind = np.argwhere(np.logical_and(
                    np.asarray(trainer.reward_value_log)[1:trainer.iteration, 0] != prev_reward_value,
                    np.asarray(trainer.executed_action_log)[1:trainer.iteration, 0] == sample_primitive_action_id)).flatten()

                if sample_ind.size > 0:

                    sample_iteration = utils.get_replay_id(trainer.predicted_value_log, trainer.label_value_log, trainer.reward_value_log, sample_ind, 'regular')
                    replay_training(sample_iteration, sample_primitive_action)

                # augment training
                if augment_training and np.random.uniform() < min(0.5, (len(trainer.augment_ids)+1)/100.0):
                    candidate_ids = trainer.augment_ids
                    try:
                        trainer.label_value_log[trainer.augment_ids[-1]]
                    except IndexError:
                        candidate_ids = trainer.augment_ids[:-1]
                    augment_replay_id = utils.get_replay_id(trainer.predicted_value_log, trainer.label_value_log, trainer.reward_value_log, candidate_ids, 'augment')
                    replay_training(augment_replay_id, 'grasp', 'augment')

                if not augment_training and len(trainer.augment_ids):
                    augment_training = True

        if not is_testing:
            if trainer.iteration % 500 == 0:
                logger.save_model(trainer.iteration, trainer.model)
                if trainer.use_cuda:
                    trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        # -------------------------------------------------------------
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        # Train coordinator
        if not is_testing:
            lc, acc = coordinator.optimize_model()
            if lc is not None:
                trainer.sync_loss.append(lc)
                trainer.sync_acc.append(acc)
                logger.write_to_log('sync-loss', trainer.sync_loss)
                logger.write_to_log('sync-acc', trainer.sync_acc)
            if trainer.iteration % 500 == 0:
                coordinator.save_networks(trainer.iteration)

        # Save information for next training step
        if not nonlocal_variables['seeking_target']:
            prev_color_img = color_img.copy()
            prev_depth_img = depth_img.copy()
            prev_color_heightmap = color_heightmap.copy()
            prev_depth_heightmap = depth_heightmap.copy()
            prev_valid_depth_heightmap = valid_depth_heightmap.copy()

            prev_mask_heightmaps = mask_heightmaps.copy()
            prev_target_mask_heightmap = target_mask_heightmap.copy()

            prev_target_grasped = nonlocal_variables['target_grasped']
            prev_primitive_action = nonlocal_variables['primitive_action']
            prev_best_pix_ind = nonlocal_variables['best_pix_ind']
            prev_push_end_pix_yx = nonlocal_variables['push_end_pix_yx']
            prev_margin_occupy_ratio = nonlocal_variables['margin_occupy_ratio']

        robot.objects_reset = False
        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    # --------------- Setup options ---------------
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234)
    parser.add_argument('--force_cpu', dest='force_cpu', action='store_true', default=False)

    # --------------- Object options ---------------
    parser.add_argument('--config_file', dest='config_file', action='store', default='simulation/random/random-3blocks.txt')

    # ------------- Algorithm options -------------
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--stage_epoch', dest='stage_epoch', type=int, action='store', default=1000)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_target_seeking', dest='test_target_seeking', action='store_true', default=False)
    parser.add_argument('--max_motion_onecase', dest='max_motion_onecase', type=int, action='store', default=20, help='maximum number of motions per test trial')
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=5, help='number of repeated test trials')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_true', default=False)
    parser.add_argument('--critic_ckpt', dest='critic_ckpt', action='store')
    parser.add_argument('--coordinator_ckpt', dest='coordinator_ckpt', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False)
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True)
    
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
