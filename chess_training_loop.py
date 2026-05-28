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
from constants import BATCH_SIZE
from agents import DQNMoveAgent, A2CMoveAgent


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
        boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, mode)
        batched_board = boards.to_tensor().cuda()
        starting_position = batched_board[0].clone()
        our_ai_agent = DQNMoveAgent(boards, starting_position, {})  # {"firepower", "firepower_per_num_moves"}
        # Recommended batch size = 256
    elif args.algorithm == "A2C":
        boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, mode)
        batched_board = boards.to_tensor().cuda()
        starting_position = batched_board[0].clone()
        our_ai_agent = A2CMoveAgent(boards, starting_position, {}, args.artifacts_dir)
        assert boards.get_batch_size() == 1
    elif args.algorithm == "A2C-MP":
        pass  # TODO.
    our_ai_agent.prepare_for_training()
    current_epoch = 0
    for i in range(0, 1):
        our_ai_agent.prepare_for_training()
        current_epoch = our_ai_agent.self_play_and_training_session(boards, current_epoch)
    our_ai_agent.save_all_models()
    our_ai_agent.save_data()
