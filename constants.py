# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:34:00 2026

@author: jledragon
"""

BATCH_SIZE = 1  # 1 for A2C, 256 for DQN
"""
How many parallel streams of chess logic we want in a Boards object.
"""

TOTAL_DESIRED_LOGGED_GAMES_A2C = 1
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