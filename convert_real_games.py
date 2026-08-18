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
from chess_py_utils import convert_UCI_to_jle_notation
from agents import A2CGameMemory
from interface import JLEAIMoveAgent
import chess_cpp


class PredestinedMoveAgent(JLEAIMoveAgent):
    """
    A move agent that has no logic of its own, but will only read and enact moves in
    pre-played games.
    """
    def __init__(self, boards, starting_position):
        self.boards = boards
        self.starting_position = starting_position

    def decide_move(self, board_state):
        pass

    def prepare_for_training(self):
        pass

    def prepare_for_evaluation(self):
        pass


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
    assert torch.cuda.is_available(), "CUDA is not enabled. Please fix this before running this script."
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
        jle_board_generator = chess_cpp.BatchedBoard(True, 1, 0)
        jle_board = jle_board_generator.to_tensor().cuda()
        agent = PredestinedMoveAgent(jle_board_generator, jle_board)
        colour_list = torch.ones((1)).to(torch.bool).cuda()
        dud_move_count = jle_board_generator.get_starting_move_count_list().cuda()
        turn = True  # For readability. True = white, False = black.

        standard_board = chess.Board()
        moves_this_game = game["moves"]
        winner = game["winner"]
        if winner == 'white':
            value = torch.Tensor([1]).to(torch.float32)
        elif winner == 'black':
            value = torch.Tensor([-1]).to(torch.float32)
        else:
            assert winner == 'draw'
            value = torch.Tensor([0]).to(torch.float32)
        converted_games_w.append(jle_board.clone())  # What white sees before they select a move

        individual_moves = moves_this_game.split(" ")
        for move in individual_moves:
            # Retain standard chess boards and JLE chess boards in lock step with another. This is so that
            # we can both translate standard notation and get the full 8-bit encoding, which we'll need
            # for other scripts.
            uci_move = standard_board.parse_san(move).uci()
            jle_move = convert_UCI_to_jle_notation(uci_move, torch.logical_not(colour_list))
            board_state = (jle_board, colour_list, dud_move_count)
            _, new_state = agent.enact_move(jle_move, board_state)
            dud_move_count, jle_board, colour_list, _, _ = new_state
            standard_board.push_san(move)
            if standard_board.is_checkmate() or standard_board.is_insufficient_material() or standard_board.is_stalemate():
                break
            if turn:
                converted_games_w.append(jle_board.clone())  # What white sees before they select a move
            else:
                converted_games_b.append(jle_board.clone())  # What black sees before they select a move
            turn = not turn

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
