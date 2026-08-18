# -*- coding: utf-8 -*-
"""
Created on Thurs May 28th 16:18:02 202

@author: jledragon

Evaluate a trained chess agent against some other agent.
Run this file with environment "jle".
"""

import torch
import argparse
from random import random
from eval_util import evaluate_agent_against_random
from chess_py_utils import flip_board, get_mode_str
from constants import BATCH_SIZE, DQN_USE_STATE_ACTIONS
import chess_cpp
from agents import StockfishMoveAgent, RandomMoveAgent, DQNMoveAgent, A2CMoveAgent, DQNExperienceBuffer, A2CGameMemory
from neural_networks import DQNChessNetwork, A2CChessNetwork


def evaluate_against_stockfish(boards, stockfish_agent, our_ai_agent):
    """
    Evaluate one of the AIs against Stockfish at some difficulty level.
    """
    boards.update_batch_size(1)

    stockfish_agent.set_elo(1_320)  # Min ELO is now 1,320!!
    for i in range(0, 20):
        stockfish_agent.start_new_game()
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((1)).to(torch.bool).cuda()
        single_board = boards.to_tensor().cuda()
        winner = ''
        if random() > 0.5:  # 0.5
            # Play white.
            starting_colour = True
            (randomly_selected_move, random_promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            stockfish_agent.log_opponent_move(randomly_selected_move, random_promotion, single_board, starting_colour)
        else:
            # Play black.
            starting_colour = False
        while True:
            stockfish_move, _ = stockfish_agent.decide_and_enact_move(single_board)
            (jle_move, jle_promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.log_stockfish_move(stockfish_move, (single_board, colour_list, dud_move_count), starting_colour)
            
            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'Stockfish'
                else:
                    winner = 'Draw'
                break

            (randomly_selected_move, random_promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            flipped_board, _ = flip_board(single_board, colour_list) #####
            stockfish_agent.log_opponent_move(randomly_selected_move, random_promotion, flipped_board, starting_colour)
            
            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'JLE'
                else:
                    winner = 'Draw'
                break
    
        print(f"I did a whole game - The winner was {winner}")

    boards.update_batch_size(BATCH_SIZE)


def evaluate_a2c_against_random(boards, random_agent, our_ai_agent):
    """
    Evaluate an A2C AI against random, to test whether training is working.
    This version uses the full MCTS with 800 simulations per turn for A2C.
    """
    boards.update_batch_size(1)

    for i in range(0, 20):
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((1)).to(torch.bool).cuda()
        single_board = boards.to_tensor().cuda()
        winner = ''
        episode_length = 0
        our_ai_agent.start_episode()
        inform_about_opponent_move = True  # When we play black, do not inform about the first move white takes.
        # This is due to starting the graph at our first playable move, which will not be the usual chess starting state.
        #col = False
        if random() > 0.5:  # 0.5
            # Play white.
            #col = True
            (move, promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            episode_length += 1
        else:
            our_ai_agent.whites_move = False
            inform_about_opponent_move = False
        while True:
            #col = not col
            (move, promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                random_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            if inform_about_opponent_move:
                our_ai_agent.log_opponent_move(move, promotion, (single_board, colour_list, dud_move_count))
            else:
                inform_about_opponent_move = True
            episode_length += 1
            if episode_length >= our_ai_agent.max_episode_length:
                winner = 'no winner. The game went on too long.'
                break

            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'Random'
                else:
                    winner = 'Draw'
                break

            #col = not col
            (move, promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            episode_length += 1
            if episode_length >= our_ai_agent.max_episode_length:
                winner = 'no winner. The game went on too long.'
                break

            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'JLE'
                else:
                    winner = 'Draw'
                break

        single_board, colour_list, dud_move_count = our_ai_agent.reset_all((single_board, colour_list, dud_move_count))
        our_ai_agent.end_episode()
        print(f"I did a whole game - The winner was {winner}")

    boards.update_batch_size(BATCH_SIZE)


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for evaluating a chess engine.')
    parser.add_argument(
        'algorithm',
        type=str,
        default="DQN",
        choices=["DQN", "A2C", "A2C-no-MCTS"],
        help="The algorithm of the model we're trying to evaluate."
    )
    parser.add_argument(
        'eval_opponent',
        type=str,
        default='random',
        choices=["random", "Stockfish"],
        help='The opponent to evaluate against.'
    )
    parser.add_argument(
        '--train_further',
        action='store_true',
        help='Whether to train the model more on the data first before evaluation.',
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
    boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, mode)
    batched_board = boards.to_tensor().cuda()
    starting_position = batched_board[0].clone()
    if args.algorithm == "DQN":
        model = DQNChessNetwork(DQN_USE_STATE_ACTIONS)
        memory = DQNExperienceBuffer(10_000, BATCH_SIZE, DQN_USE_STATE_ACTIONS)
        our_ai_agent = DQNMoveAgent(boards, model, memory, starting_position, {}, DQN_USE_STATE_ACTIONS)  # {"firepower", "firepower_per_num_moves"}
        assert boards.get_batch_size() == 256
    elif args.algorithm == "A2C":
        model = A2CChessNetwork()
        memory = A2CGameMemory(256)
        our_ai_agent = A2CMoveAgent(boards, model, starting_position, {}, True)
        assert boards.get_batch_size() == 1
    elif args.algorithm == "A2C-no-MCTS":
        model = A2CChessNetwork()
        memory = A2CGameMemory(256)
        our_ai_agent = A2CMoveAgent(boards, model, starting_position, {}, False)
        assert boards.get_batch_size() == 256
    if args.train_further:
        memory.load_data('latest')
        model.load_models('train')
        #our_ai_agent.analyse_loaded_data_and_models()  # Debugging
        our_ai_agent.train_on_loaded_data(memory, 'latest')
    else:
        model.load_models('eval')
    our_ai_agent.prepare_for_evaluation()
    if args.eval_opponent == 'Stockfish' and args.algorithm == 'A2C-no-MCTS':
        raise NotImplementedError("Playing Stockfish without MCTS against Stockfish is not yet implemented.")
    elif args.eval_opponent == 'Stockfish':
        # TODO - allow any mode for Stockfish
        stockfish_agent = StockfishMoveAgent(boards, starting_position)
        evaluate_against_stockfish(boards, stockfish_agent, our_ai_agent)
    elif args.eval_opponent == 'random' and args.algorithm == 'DQN':
        random_agent = RandomMoveAgent(boards, starting_position)
        evaluate_agent_against_random(boards, random_agent, our_ai_agent)
    elif args.eval_opponent == 'random' and args.algorithm == 'A2C':
        random_agent = RandomMoveAgent(boards, starting_position)
        evaluate_a2c_against_random(boards, random_agent, our_ai_agent)
    elif args.eval_opponent == 'random' and args.algorithm == 'A2C-no-MCTS':
        random_agent = RandomMoveAgent(boards, starting_position)
        evaluate_agent_against_random(boards, random_agent, our_ai_agent)
