# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:24:34 2026

@author: jledragon
"""

import torch
import chess_cpp
from abc import ABC, abstractmethod
from chess_py_utils import (
    conditional_compile,
    convert_UCI_to_jle_notation,
    flip_board,
    get_repetition_status,
    get_white_view,
    is_game_over,
    possibly_reset_game,
    reset_colour_list,
    reset_move_counts,
)

class AIMoveAgent(ABC):
    """
    Any and all autonomous agents should inherit this interface.
    """

    @abstractmethod
    def decide_move(self, board_state):
        """
        Choose a move given the board state.
        """
        pass
    
    @abstractmethod
    def enact_move(self, move, board_state):
        """
        Play the chosen move forward
        """
        pass

    def decide_and_enact_move(self, board_state):
        """
        Play a full turn.
        """
        move = self.decide_move(board_state)
        move, state = self.enact_move(move, board_state)
        return move, state


class JLEAIMoveAgent(AIMoveAgent, ABC):
    """
    An AI that can play against Stockfish.
    """
    
    @abstractmethod
    def prepare_for_training(self):
        pass
    
    @abstractmethod
    def prepare_for_evaluation(self):
        pass

    def log_stockfish_move(self, move, board_state, starting_colour_me):
        """
        Record to the JLE game state what the Stockfish move was.
        """
        jle_move, jle_promotion = convert_UCI_to_jle_notation(move, starting_colour_me)
        return self.enact_move((jle_move, jle_promotion), board_state)
    
    @conditional_compile
    def enact_move(self, move, board_state, full_game_over_conditions=True):
        # If full_game_over_conditions is False, skip ongoing draw conditions such as 50-move rule and threefold repetition.
        # Some algorithms need to explore, so these should be disabled during exploration.
        selected_move, promotion = move
        board_tensor, colour_list, dud_move_count = board_state
        if not full_game_over_conditions:
            dud_move_count = torch.zeros((board_tensor.shape[0], 1)).to(board_tensor.device).to(torch.int8)
        # Do the move
        chess_cpp.enact_moves(board_tensor, selected_move, promotion, dud_move_count)
        # Flip the board to opponent view
        flipped_board, inv_colour_list = flip_board(board_tensor, colour_list)
        # Log this position (white's view) for threefold repetition check
        white_view = get_white_view(board_tensor, flipped_board, colour_list)
        if full_game_over_conditions:
            repetition_status = get_repetition_status(self.boards, white_view)
        else:
            repetition_status = torch.zeros((board_tensor.shape[0], 1)).to(board_tensor.device).to(torch.bool)
        colour_list = inv_colour_list
        # Get opponent's move to see if the game is over by the move we just did
        opponent_move_layer = chess_cpp.get_moves_for_player(flipped_board)
        # Check whether the game is over under any condition
        game_over_tensor = is_game_over(flipped_board, opponent_move_layer, repetition_status, dud_move_count)
        game_over = torch.any(game_over_tensor, dim=1)
        # Reset threefold repetitions list, move counts and colour wherever the game is over
        if full_game_over_conditions:
            self.boards.reset_repetitions(game_over)
        reset_move_counts(dud_move_count, game_over)
        reset_colour_list(colour_list, game_over)
        # Flip the boards, reset wherever the game is over
        board_tensor = flipped_board
        board_tensor = possibly_reset_game(board_tensor, game_over, self.starting_position)
        
        return (selected_move, promotion), (dud_move_count, board_tensor, colour_list, opponent_move_layer, game_over_tensor)

    @conditional_compile
    def reset_all(self, board_state):
        board_tensor, colour_list, dud_move_count = board_state
        game_over_all = torch.ones((board_tensor.shape[0])).cuda().to(torch.bool)
        self.boards.reset_repetitions(game_over_all)
        reset_move_counts(dud_move_count, game_over_all)
        reset_colour_list(colour_list, game_over_all)
        board_tensor = possibly_reset_game(board_tensor, game_over_all, self.starting_position)
        return board_tensor, colour_list, dud_move_count
