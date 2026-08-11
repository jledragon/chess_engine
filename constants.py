# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:34:00 2026

@author: jledragon
"""

BATCH_SIZE = 1  # 1 for A2C, 256 for DQN
"""
How many parallel streams of chess logic we want in a Boards object.
"""

TOTAL_DESIRED_LOGGED_GAMES_A2C = 100
"""
The point at which to stop training A2C.
"""

A2C_TRAIN_CADENCE = 20
"""
How many more games do we need before we start training the A2C network from the training memory?
"""

MCTS_BATCH_SIZE = 256
"""
The max. batch size to be used for MCTS.
"""

TRAINING_BATCH_SIZE = 256
"""
When training the neural network, the batch size to use.
"""

A2C_TRAINING_ITERS = 400
"""
How many steps to train for during A2C's training phase.
"""

A2C_TRAINING_DELTA = 200
"""
How much to increase the number of training steps for A2C after more data comes in.
"""

DQN_USE_STATE_ACTIONS = False
"""
Indicates which of the two modes to run DQN in. These affect what a 'state' is:

False - A state is the board only. The neural network returns a value for every possible action.
True - A state is the board + the action. The neural network returns a single value.
    Every state/action combo should be run through the network.
"""
