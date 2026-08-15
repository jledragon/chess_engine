# -*- coding: utf-8 -*-
"""
Created on Sun Nov 5th 18:39:05 2023

@author: jledragon

Train chess engine agents through self-play.
Run this file with environment "jle".
"""

import torch
import chess_cpp
import argparse
from chess_py_utils import get_mode_str
from constants import BATCH_SIZE, TRAINING_BATCH_SIZE, DQN_USE_STATE_ACTIONS
from agents import DQNExperienceBuffer, A2CGameMemory, train_single_threaded_a2c, train_multi_threaded_a2c, train_dqn
from neural_networks import DQNChessNetwork, A2CChessNetwork
from datetime import datetime as dt
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for training a chess engine.')
    parser.add_argument(
        'algorithm',
        type=str,
        default="DQN",
        choices=["DQN", "A2C", "A2C-MP"],
        help='The algorithm to use for training.'
    )
    parser.add_argument(
        '--artifacts_dir',
        type=str,
        default='artifacts',
        help='The folder to save artifacts in, starting from the present directory.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Starting condition
    args = get_args()
    # Insist that we have CUDA for now, otherwise things will be much slower.
    assert torch.cuda.is_available(), "CUDA is not enabled. Please fix this before running this script."
    torch._dynamo.config.cache_size_limit = 64
    mode = 1
    mode_str = get_mode_str(mode)
    print(f"Training with {mode_str} chess games")
    if args.algorithm == "DQN":
        model = DQNChessNetwork(DQN_USE_STATE_ACTIONS)
        memory = DQNExperienceBuffer(10_000_000, BATCH_SIZE, DQN_USE_STATE_ACTIONS)
        boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, mode)
        batched_board = boards.to_tensor().cuda()
        starting_position = batched_board[0].clone()
        train_dqn(args, model, memory, mode, DQN_USE_STATE_ACTIONS)
        # Recommended batch size = 256
    elif args.algorithm == "A2C":
        model = A2CChessNetwork()
        memory = A2CGameMemory(TRAINING_BATCH_SIZE)
        now_str = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifacts_dir = Path('.') / args.artifacts_dir / now_str
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        train_single_threaded_a2c(artifacts_dir, model, memory, mode)
    elif args.algorithm == "A2C-MP":
        # To kill all processes mid-flight, run taskkill /im python.exe /f /t
        model = A2CChessNetwork()
        memory = A2CGameMemory(TRAINING_BATCH_SIZE)
        now_str = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifacts_dir = Path('.') / args.artifacts_dir / now_str
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        train_multi_threaded_a2c(artifacts_dir, model, memory, mode)
    model.save_models()
    memory.save_data()
