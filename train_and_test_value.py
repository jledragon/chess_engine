# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:20:53 2026

Train and test only the value part of A2C.

@author: jledragon
"""

import argparse
import torch
from agents import A2CMoveAgent, A2CGameMemory
from neural_networks import A2CChessNetwork
import chess_cpp


def get_args():
    parser = argparse.ArgumentParser(description='Parameters training and testing the value part of A2C.')
    parser.add_argument(
        'train_set',
        type=str,
        help="The name of the file that contains the training data."
    )
    parser.add_argument(
        'test_set',
        type=str,
        help='The name of the file that contains the test data.'
    )
    parser.add_argument(
        'batch_size',
        type=int,
        default=256,
        help='Batch size for training.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    assert torch.cuda.is_available(), "CUDA is not enabled. Please fix this before running this script."
    boards = chess_cpp.BatchedBoard(True, 1, 0)
    batched_board = boards.to_tensor().cuda()
    starting_position = batched_board[0].clone()
    torch._dynamo.config.cache_size_limit = 64
    model = A2CChessNetwork()
    memory = A2CGameMemory(args.batch_size, val_only_mode=True)
    our_ai_agent = A2CMoveAgent(boards, model, starting_position, {}, False)

    # train
    our_ai_agent.prepare_for_training()
    memory.load_data(args.train_set)
    our_ai_agent = A2CMoveAgent(boards, model, starting_position, {}, False)
    our_ai_agent.train_on_loaded_data(memory)
    print("Eval metrics on training data:")
    our_ai_agent.eval_val_on_loaded_data(memory)

    # test
    our_ai_agent.prepare_for_evaluation()
    memory.load_data(args.test_set)
    print("Eval metrics on test data:")
    our_ai_agent.eval_val_on_loaded_data(memory)
