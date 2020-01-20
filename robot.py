import time
import os
import numpy as np
import utils
from simulation import vrep


class Robot(object):
    def __init__(self, workspace_limits, is_testing, test_preset_cases, config_file):

        self.workspace_limits = workspace_limits

        # If in simulation...
        self.obj_mesh_dir = os.path.abspath('./simulation/blocks')
        self.mesh_list = [f for f in os.listdir(self.obj_mesh_dir) if f.endswith('.obj')]

        # Define colors for object meshes (Tableau palette)
        self.color_labels = ['block', 'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        self.color_space = np.asarray([[184, 179, 168],
                                       [255, 0, 0],
                                       [255, 127, 0],
                                       [255, 255, 0],
                                       [0, 255, 0],
                                       [0, 0, 255],
                                       [38, 0, 51],
                                       [148, 0, 211]])/255.0

        # Make sure to have the server side running in V-REP:
        # in a child script of a V-REP scene, add following command
        # to be executed just once, at simulation start:
        #
        # simExtRemoteApiStart(19999)
        #
        # then start simulation, and run this program.
        #
        # IMPORTANT: for each successful call to simxStart, there
        # should be a corresponding call to simxFinish at the end!

        # MODIFY remoteApiConnections.txt

        # Connect to simulator
        vrep.simxFinish(-1)  # Just in case, close all opened connections
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        self.is_testing = is_testing
        self.test_preset_cases = test_preset_cases

        # Setup virtual camera in simulation
        self.setup_sim_camera()

        # Read config file
        self.sim_read_config_file(config_file)

        # Add objects to simulation environment
        self.objects_reset = False
        self.add_objects()

    def sim_read_config_file(self, config_file):
        if not os.path.isfile(config_file):
            raise Exception('config file not exist')

        self.config_file = config_file

        # If testing, read object names and poses from test case file; else just read names
        self.obj_labels = []
        if self.is_testing and self.test_preset_cases:
            self.obj_mesh_files = []
            self.obj_positions = []
            self.obj_orientations = []
        with open(self.config_file, 'r') as file:
            lines = (line.rstrip() for line in file)  # All lines including the blank ones
            lines = (line for line in lines if line)  # Non-blank lines
            for line in lines:
                line_content = line.split()
                if line_content[0].startswith('#'):
                    continue
                self.obj_labels.append(line_content[0])
                if self.is_testing and self.test_preset_cases:
                    self.obj_mesh_files.append(line_content[1])
                    self.obj_positions.append([float(line_content[2]), float(line_content[3]), float(line_content[4])])
                    self.obj_orientations.append(
                        [float(line_content[5]), float(line_content[6]), float(line_content[7])])

        self.num_obj = len(self.obj_labels)

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_ortho',
                                                            vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1,
                                                           vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1,
                                                                 vrep.simx_opmode_blocking)
        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4, 4)
        cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

    def add_objects(self):
        self.objects_reset = True

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        self.object_handles = []
        self.object_names = []

        c_blocks = 0
        for object_idx in range(self.num_obj):
            obj_label = self.obj_labels[object_idx]
            color_id = self.color_labels.index(obj_label)
            object_color = [self.color_space[color_id][0], self.color_space[color_id][1],
                            self.color_space[color_id][2]]
            if obj_label == 'block':
                c_blocks += 1
                obj_label = obj_label + str(c_blocks)
            self.object_names.append(obj_label)
            if self.is_testing and self.test_preset_cases:
                mesh_file = os.path.join(self.obj_mesh_dir, self.obj_mesh_files[object_idx])
                object_position = [self.obj_positions[object_idx][0], self.obj_positions[object_idx][1],
                                   self.obj_positions[object_idx][2]]
                object_orientation = [self.obj_orientations[object_idx][0],
                                      self.obj_orientations[object_idx][1],
                                      self.obj_orientations[object_idx][2]]
            else:
                mesh_file = os.path.join(self.obj_mesh_dir, np.random.choice(self.mesh_list))
                drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + \
                          self.workspace_limits[0][0] + 0.1
                drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + \
                          self.workspace_limits[1][0] + 0.1
                object_position = [drop_x, drop_y, 0.1]
                object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                      2 * np.pi * np.random.random_sample()]

            ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                                  'remoteApiCommandServer',
                                                                                                  vrep.sim_scripttype_childscript,
                                                                                                  'importShape',
                                                                                                  [0, 0, 255, 0],
                                                                                                  object_color + object_position + object_orientation,
                                                                                                  [mesh_file, obj_label],
                                                                                                  bytearray(),
                                                                                                  vrep.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            self.object_handles.append(ret_ints[0])
            if not self.is_testing:
                time.sleep(2)

    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5, 0, 0.3),
                                   vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(0.5)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(0.5)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                                   vrep.simx_opmode_blocking)

    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < \
                 self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and \
                 gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > \
                 self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()

    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0],  # red
                                    [-0.625, -0.125, 0.0],  # blue
                                    [-0.375, 0.125, 0.0],  # green
                                    [-0.375, -0.125, 0.0]])  # yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1, obj_positions.shape[1], 1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)

    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached

    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1,
                                                                  vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_pixels(self):
        positions = np.array(self.get_obj_positions())
        uvw = np.matmul(self.cam_intrinsics, np.matmul(np.transpose(self.cam_pose[0:3, 0:3]),
                                                       np.transpose(positions) - np.reshape(self.cam_pose[0:3,3], [3,1])))
        uv = uvw[0:2, :]/uvw[2, :]
        return np.transpose(uv)

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1,
                                                                  vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1,
                                                                        vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:
            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + \
                     workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + \
                     workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                  2 * np.pi * np.random.random_sample()]
            vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation,
                                          vrep.simx_opmode_blocking)
            time.sleep(2)

    def get_camera_data(self):

        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0,
                                                                       vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle,
                                                                                vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img

    def close_gripper(self):

        gripper_motor_velocity = -0.5
        gripper_motor_force = 100
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                               vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                    vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)
        gripper_fully_closed = False
        while gripper_joint_position > -0.047:  # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                            vrep.simx_opmode_blocking)
            # print(gripper_joint_position)
            if new_gripper_joint_position >= gripper_joint_position:
                return gripper_fully_closed
            gripper_joint_position = new_gripper_joint_position
        gripper_fully_closed = True

        return gripper_fully_closed

    def open_gripper(self):

        gripper_motor_velocity = 0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                               vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                    vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)
        while gripper_joint_position < 0.0536:  # Block until gripper is fully open
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                        vrep.simx_opmode_blocking)

    def move_to(self, tool_position, tool_orientation):

        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)

        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02 * move_direction / move_magnitude
        num_move_steps = int(np.floor(move_magnitude / 0.02))

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
            UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1],
            UR5_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                      vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]),
                                   vrep.simx_opmode_blocking)

    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

        # Move gripper to location above grasp target
        grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05 * move_direction / move_magnitude + 1e-8
        num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                     vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
            UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
            UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
            UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
            np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                          vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]),
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                      (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

        # Ensure gripper is open
        self.open_gripper()

        # Approach grasp target
        self.move_to(position, None)

        # Close gripper to grasp target
        gripper_full_closed = self.close_gripper()

        # Move gripper to location above grasp target
        self.move_to(location_above_grasp_target, None)

        # Check if grasp is successful
        gripper_full_closed = self.close_gripper()
        grasp_success = not gripper_full_closed

        # Move the grasped object elsewhere
        grasped_object_name = None
        if grasp_success:
            object_positions = np.asarray(self.get_obj_positions())
            object_positions = object_positions[:, 2]
            grasped_object_ind = np.argmax(object_positions)
            grasped_object_handle = self.object_handles[grasped_object_ind]
            grasped_object_name = self.object_names[grasped_object_ind]
            del self.object_handles[grasped_object_ind]
            del self.object_names[grasped_object_ind]
            vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1,
                                       (-1, -1 + 0.05 * float(grasped_object_ind), 0.1),
                                       vrep.simx_opmode_blocking)
        return grasped_object_name

    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

        # Adjust pushing point to be on tip of finger
        position[2] = position[2] + 0.015

        # Compute pushing direction
        push_orientation = [1.0, 0.0]
        push_direction = np.asarray([push_orientation[0] * np.cos(heightmap_rotation_angle) - push_orientation[1] * np.sin(heightmap_rotation_angle),
                                     push_orientation[0] * np.sin(heightmap_rotation_angle) + push_orientation[1] * np.cos(heightmap_rotation_angle)])

        # Move gripper to location above pushing point
        pushing_point_margin = 0.1
        location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_pushing_point
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        move_direction = np.asarray(
            [tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],
             tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05 * move_direction / move_magnitude + 1e-8
        num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                                                     vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
            UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
            UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
            UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
            np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
                                          vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]),
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
                                      (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

        # Ensure gripper is closed
        self.close_gripper()

        # Approach pushing point
        self.move_to(position, None)

        # Compute target location (push to the right)
        push_length = 0.1
        target_x = min(max(position[0] + push_direction[0] * push_length, workspace_limits[0][0]),
                       workspace_limits[0][1])
        target_y = min(max(position[1] + push_direction[1] * push_length, workspace_limits[1][0]),
                       workspace_limits[1][1])
        push_length = np.sqrt(np.power(target_x - position[0], 2) + np.power(target_y - position[1], 2))

        # Move in pushing direction towards target location
        self.move_to([target_x, target_y, position[2]], None)

        # Move gripper to location above grasp target
        self.move_to([target_x, target_y, location_above_pushing_point[2]], None)
