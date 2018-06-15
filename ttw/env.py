# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy

def step_agnostic(action, loc, boundaries):
    """Return new location after """
    new_loc = copy.deepcopy(loc)
    step = {'UP': (0, 1), 'RIGHT': (1, 0), 'DOWN': (0, -1), 'LEFT': (-1, 0)}[action]
    new_loc[0] = min(max(loc[0] + step[0], boundaries[0]), boundaries[2])
    new_loc[1] = min(max(loc[1] + step[1], boundaries[1]), boundaries[3])
    return new_loc


def step_aware(action, loc, boundaries):
    orientations = ['N', 'E', 'S', 'W']
    steps = dict()
    steps['N'] = [0, 1]
    steps['E'] = [1, 0]
    steps['S'] = [0, -1]
    steps['W'] = [-1, 0]

    new_loc = copy.deepcopy(loc)
    if action == 'ACTION:TURNLEFT':
        # turn left
        new_loc[2] = (new_loc[2] - 1) % 4

    if action == 'ACTION:TURNRIGHT':
        # turn right
        new_loc[2] = (new_loc[2] + 1) % 4

    if action == 'ACTION:FORWARD':
        # move forward
        orientation = orientations[loc[2]]
        new_loc[0] = new_loc[0] + steps[orientation][0]
        new_loc[1] = new_loc[1] + steps[orientation][1]

        new_loc[0] = min(max(new_loc[0], boundaries[0]), boundaries[2])
        new_loc[1] = min(max(new_loc[1], boundaries[1]), boundaries[3])
    return new_loc