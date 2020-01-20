#!/usr/bin/env python

import os
import argparse
import numpy as np

# Parse session directories
parser = argparse.ArgumentParser()
parser.add_argument('--session_directory', dest='session_directory', action='store', type=str, help='path to session directory for which to measure performance')
args = parser.parse_args()
session_directory = args.session_directory

# Parse data from session (reposition log, reward values log)
transitions_directory = os.path.join(session_directory, 'transitions')
target_grasped_log = np.loadtxt(os.path.join(transitions_directory, 'target-grasped.log.txt'), delimiter=' ').astype(int)
reposition_log = np.loadtxt(os.path.join(transitions_directory, 'reposition.log.txt'), delimiter=' ').astype(int)

max_num_motion = 20

num_trials = len(reposition_log)
result_rec = []
num_motions_rec = []

for i in range(num_trials):
    if i == 0:
        result = reposition_log[i] <= max_num_motion and target_grasped_log[reposition_log[i]-1]
        result_rec.append(result)
        num_motions = min(reposition_log[i], max_num_motion)
        num_motions_rec.append(num_motions)
    else:
        result = (reposition_log[i]-reposition_log[i-1]) <= max_num_motion and target_grasped_log[reposition_log[i]-1]
        result_rec.append(result)
        num_motions = min(reposition_log[i]-reposition_log[i-1], max_num_motion)
        num_motions_rec.append(num_motions)

# Display results
print('Success rate %0.1f' % float(np.mean(result_rec)*100))
print('Mean number of motions %0.2f, std %0.2f' % (float(np.mean(num_motions_rec)), float(np.std(num_motions_rec))))
