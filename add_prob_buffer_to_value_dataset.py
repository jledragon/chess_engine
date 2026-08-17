# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:07:46 2026

For a dataset that has values only, such as one created with convert_real_games.py,
Compute MCTS per game from a fresh tree and save the probability buffer to the
dataset. Save it as a new file with a new name.

@author: jledragon
"""

import argparse
import torch
import chess_cpp
from agents import A2CMoveAgent, A2CGameMemory, MCTSGraph
from neural_networks import A2CChessNetwork


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for adding the probability buffer to value datasets.')
    parser.add_argument(
        'file_name',
        type=str,
        help="The name of the file that contains the data."
    )
    parser.add_argument(
        '-mcts_depth',
        type=int,
        default=800,
        help='Number of simulations per run of MCTS.'
    )
    parser.add_argument(
        '--disallow_noise',
        action='store_true',
        help='Whether to prevent Dirichlet noise being added when running MCTS simulations.',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    assert torch.cuda.is_available(), "CUDA is not enabled. Please fix this before running this script."
    boards = chess_cpp.BatchedBoard(True, 1, 0)
    torch._dynamo.config.cache_size_limit = 64
    model = A2CChessNetwork()
    memory = A2CGameMemory(256, val_only_mode=True)
    memory.load_data(args.file_name)

    for i in range(memory.state_buffer.shape[0]):
        board_i = memory.state_buffer[i:i+1,:,:,:].cuda()
        print(board_i.shape)
        our_ai_agent = A2CMoveAgent(boards, model, board_i, {}, False)
        white_mcts = MCTSGraph(our_ai_agent, boards)
        move_layer = chess_cpp.get_moves_for_player(board_i)
        white_mcts.init_top_node_if_empty_graph(board_i, move_layer)
        white_mcts.generate_graph(not args.disallow_noise)  # Allow some noise to be added.
        assert False
