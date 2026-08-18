# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:19:43 2026

@author: jledragon
"""

import torch
from chess_py_utils import conditional_compile


@conditional_compile
def evaluate_agent_against_random(boards, random_agent, our_ai_agent):
    """
    Play an AI against random, to test whether a new training algorithm is working.

    - With DQN, this just works with DQN as it is.
    - With A2C, this works with a non-MCTS version, as running A2C with MCTS would be
        intractible with 800 simulations per move. Here, we just select the best action
        directly from one network forward pass.
    """
    boards.update_batch_size(256)
    batched_board = boards.to_tensor().cuda()
    dud_move_count = boards.get_starting_move_count_list()
    colour_list = torch.ones((batched_board.shape[0])).to(torch.bool).cuda()
    win_count = 0
    lose_count = 0
    draw_count = 0
    games_played = 0
    while games_played < 1000:
        # The trained AI always starts white, for the sake of argument.
        (move, promotion), (dud_move_count, batched_board, colour_list, _, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        games_won = torch.sum(game_over_tensor[:,0].to(torch.int8))
        games_drawn = torch.sum(game_over_tensor[:,1:].to(torch.int8))
        win_count += games_won
        draw_count += games_drawn
        games_played += (games_won + games_drawn)
        
        (move, promotion), (dud_move_count, batched_board, colour_list, _, game_over_tensor) = \
            random_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        games_lost = torch.sum(game_over_tensor[:,0].to(torch.int8))
        games_drawn = torch.sum(game_over_tensor[:,1:].to(torch.int8))
        lose_count += games_lost
        draw_count += games_drawn
        games_played += (games_lost + games_drawn)
        #print(games_played)
    print(f"Win count: {win_count} ({win_count / games_played * 100}%), "
          f"Lose count: {lose_count} ({lose_count / games_played * 100}%), "
          f"Draw count: {draw_count} ({draw_count / games_played * 100}%)")
