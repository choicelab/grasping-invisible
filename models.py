#!/usr/bin/env python

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from networks import FeatureTunk


class reinforcement_net(nn.Module):

    def __init__(self, use_cuda):
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.feature_tunk = FeatureTunk()

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(1024)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(128)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=False)),
            ('push-norm2', nn.BatchNorm2d(32)),
            ('push-relu2', nn.ReLU(inplace=True)),
            ('push-conv2', nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(1024)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(128)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm2', nn.BatchNorm2d(32)),
            ('grasp-relu2', nn.ReLU(inplace=True)),
            ('grasp-conv2', nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, input_mask_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                    # NOTES: affine_grid + grid_sample -> spatial transformer networks
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data).cuda(), flow_grid_before)
                        rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before)
                        rotate_mask = F.grid_sample(Variable(input_mask_data).cuda(), flow_grid_before)
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data), flow_grid_before)
                        rotate_depth = F.grid_sample(Variable(input_depth_data), flow_grid_before)
                        rotate_mask = F.grid_sample(Variable(input_mask_data), flow_grid_before)

                    # Compute intermediate features
                    interm_feat = self.feature_tunk(rotate_color, rotate_depth, rotate_mask)

                    # Forward pass through branches
                    push_out = self.pushnet(interm_feat)
                    grasp_out = self.graspnet(interm_feat)

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), push_out.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), grasp_out.data.size())

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([F.interpolate(F.grid_sample(push_out, flow_grid_after), scale_factor=16, mode='bilinear', align_corners=True),
                                        F.interpolate(F.grid_sample(grasp_out, flow_grid_after), scale_factor=16, mode='bilinear', align_corners=True)])

            return output_prob, interm_feat

        else:
            self.output_prob = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
            
            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())
             
            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before)
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before)
                rotate_mask = F.grid_sample(Variable(input_mask_data, requires_grad=False).cuda(), flow_grid_before)
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before)
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before)
                rotate_mask = F.grid_sample(Variable(input_mask_data, requires_grad=False), flow_grid_before)

            # Compute intermediate features
            self.interm_feat = self.feature_tunk(rotate_color, rotate_depth, rotate_mask)

            # Forward pass through branches
            push_out = self.pushnet(self.interm_feat)
            grasp_out = self.graspnet(self.interm_feat)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), push_out.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), push_out.data.size())
            
            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([F.interpolate(F.grid_sample(push_out, flow_grid_after), scale_factor=16, mode='bilinear', align_corners=True),
                                     F.interpolate(F.grid_sample(grasp_out, flow_grid_after), scale_factor=16, mode='bilinear', align_corners=True)])
                
            return self.output_prob, self.interm_feat
