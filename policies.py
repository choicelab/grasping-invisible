import os
import random
from collections import deque
from collections import namedtuple

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from scipy import signal
from scipy.ndimage.interpolation import shift

Transition = namedtuple('Transition', ('inputs', 'labels'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def get_data(self):
        transitions = self.memory
        data = Transition(*zip(*transitions))
        return data.inputs, data.labels

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(self.bn1(x)))
        x = F.relu(self.fc2(self.bn2(x)))
        x = self.fc3(self.bn3(x))
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Coordinator(object):

    def __init__(self, save_dir, ckpt_file, feat_size=5, buffer_size=200, batch_size=20):
        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayMemory(buffer_size)
        self.net = MLP(in_dim=feat_size)
        self.net.apply(init_weights)
        if ckpt_file is not None:
            self.load_networks(ckpt_file)
            print('Pre-trained coordinator model loaded from: %s' % ckpt_file)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None, None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        self.net.train()

        self.optimizer.zero_grad()
        outputs = self.net(torch.tensor(batch.inputs, dtype=torch.float).float())
        labels = torch.tensor(batch.labels).float().flatten().long()
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        lc = loss.cpu().detach().numpy()
        acc = self.get_accuracy()

        print('Coordinator training loss %f, acc %f' % (lc, acc))

        return lc, acc

    def predict(self, X):
        self.net.eval()
        net_outputs = self.net(torch.tensor(X, dtype=torch.float).view(-1, self.net.in_dim))
        return np.argmax(net_outputs.view(-1,2).cpu().detach().numpy(), axis=1)

    def get_accuracy(self):
        X_val, y_val = self.memory.get_data()
        y_pre = self.predict(X_val)
        accuracy = np.array(y_val == y_pre).mean()
        return accuracy

    def save_networks(self, which_epoch):
        save_filename = 'coordinator-%06d.pth' % which_epoch
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.net.cpu().state_dict(), save_path)

    def load_networks(self, load_path):
        self.net.load_state_dict(torch.load(load_path))


def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


class Explorer(object):

    def __init__(self, map_size, buffer_size=3, prob_scaled=0.75, std=25):
        # assume the map is square, so map_size is a scalar
        # prob_scaled and std is the fine-tuned parameters for our workspace

        # compute basic kernel
        gkernel = gkern(kernlen=map_size, std=std)
        self.kcenter = np.array(np.unravel_index(np.argmax(gkernel), gkernel.shape))
        ad_gkernel = (1 - prob_scaled) * gkernel / (np.max(gkernel))
        self.bkernel = 1 - ad_gkernel
        self.ones_kernel = np.ones((map_size, map_size))

        # kernel buffer
        self.kbuffer = deque([], buffer_size)
        self.kbuffer.append(self.ones_kernel)

    def get_kernel(self, center):
        bkernel_shifted = shift(self.bkernel, np.array(center).reshape(-1)-self.kcenter, cval=1.0)
        return bkernel_shifted

    def update(self, prev_act_pos):
        prev_kernel = self.get_kernel(prev_act_pos)
        self.kbuffer.append(self.ones_kernel)
        for i in range(len(self.kbuffer)):
            self.kbuffer[i] = np.multiply(self.kbuffer[i], prev_kernel)

    def reset(self):
        self.kbuffer.clear()
        self.kbuffer.append(self.bkernel)

    def get_action_maps(self, prior):
        post = np.multiply(prior, self.kbuffer[0])
        return post / np.max(post)
