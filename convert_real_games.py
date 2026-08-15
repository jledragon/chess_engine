# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:20:15 2026

Turn the real game dataset into game states that can be learned with A2C.
Here, store value, state and num. times seen only (not probs per move).
Probs per move should be computed in a different script that will run MCTS.

Specifically works with this dataset: https://www.kaggle.com/datasets/datasnaek/chess

@author: jledragon
"""

import torch
import pandas as pd
import chess
import argparse
from chess_py_utils import convert_fen_to_jle_board
from agents import A2CGameMemory


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for creating datasets from real games.')
    parser.add_argument(
        'dataset_name',
        type=str,
        default="real_games_values_train",
        help="The name of the dataset file, once saved."
    )
    parser.add_argument(
        'start',
        type=int,
        default=0,
        help='The starting point to read the dataset from.'
    )
    parser.add_argument(
        'end',
        type=int,
        default=1_000,
        help='The end point (offset from the start) where to finish reading a dataset.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    all_games = pd.read_csv("real_games/games.csv")
    valid_games = all_games[(all_games["victory_status"] == 'mate') | (all_games["insufficient material"] == True) | (all_games["stalemate"] == True)]
    print(f"Seen {valid_games.shape[0]} valid games to work with.")
    if args.start > valid_games.shape[0]:
        raise ValueError("Starting index is greater than the dataset size.")
    if args.start + args.end > valid_games.shape[0]:
        print(f"Warning - This process will gain fewer than {args.end} games.")
    memory = A2CGameMemory(256)
    for tot, (index, game) in enumerate(valid_games.iloc[args.start:].iterrows()):
        converted_games_w = []
        converted_games_b = []
        turn = True # white
        standard_board = chess.Board()
        fen_str = standard_board.fen()
        jle_board = convert_fen_to_jle_board(fen_str)
        moves_this_game = game["moves"]
        winner = game["winner"]
        if winner == 'white':
            value = torch.Tensor([1]).to(torch.float32)
        elif winner == 'black':
            value = torch.Tensor([-1]).to(torch.float32)
        else:
            assert winner == 'draw'
            value = torch.Tensor([0]).to(torch.float32)
        converted_games_w.append(jle_board)

        individual_moves = moves_this_game.split(" ")
        for move in individual_moves:
            turn = not turn
            standard_board.push_san(move)
            fen_str = standard_board.fen()
            jle_board = convert_fen_to_jle_board(fen_str)
            if turn:
                converted_games_w.append(jle_board)
            else:
                converted_games_b.append(jle_board)

        converted_games_w = torch.cat(converted_games_w)
        converted_games_b = torch.cat(converted_games_b)
        dummy_w = [torch.ones(1) for _ in range(converted_games_w.shape[0])]
        dummy_b = [torch.ones(1) for _ in range(converted_games_b.shape[0])]
        memory.add_to_memory(converted_games_w, dummy_w, value)
        memory.add_to_memory(converted_games_b, dummy_b, -value)
        if tot >= args.end:
            break

    memory.regenerate_data()
    data_map = {
        "state_buffer": memory.state_buffer,
        "game_value_buffer": memory.game_value_buffer,
        "num_times_seen": memory.num_times_seen,
    }
    torch.save(data_map, f"datasets/{args.dataset_name}.pt")
