# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:43:25 2024

@author: jledragon
"""

# Run this with "python -m unittest test_rewards.py"
import torch
import unittest
import chess_cpp
from chess_training_loop import DQNMoveAgent
from chess_py_utils import get_human_readable_board, flip_board


BATCH_SIZE = 1


class DummyDQNMoveAgent(DQNMoveAgent):
    def __init__(self, boards, starting_position, enabled_optional_rewards):
        super(DummyDQNMoveAgent, self).__init__(boards, starting_position, enabled_optional_rewards)
        self._stored_move = None

    def store_move(self, move_bundle):
        self._stored_move = move_bundle

    def decide_move(self, board_state):
        move_bundle = self._stored_move
        if move_bundle is None:
            raise ValueError("Must call store_move before each call to the test agent.")
        self._stored_move = None
        return move_bundle


class TestRewards(unittest.TestCase):
    '''
    Unit tests for expected rewards as the game is played forward.
    '''
    
    def test_both_sides_movement(self):
        '''
        Test expected rewards with no endgames.
        Assumes the following rewards are active:
            - Win/lose
            - Movement penalty
            - Maximising our number of moves/minimising opponent moves
        '''
        # Set up the board.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        my_king = chess_cpp.Piece('k', True, 0, 0)
        guard_1 = chess_cpp.Piece('p', True, 1, 0)
        guard_2 = chess_cpp.Piece('p', True, 1, 1)
        guard_3 = chess_cpp.Piece('r', True, 0, 1)
        extra_1 = chess_cpp.Piece('q', True, 1, 3)
        enemy_king = chess_cpp.Piece('k', False, 7, 7)
        guard_4 = chess_cpp.Piece('p', False, 6, 7)
        guard_5 = chess_cpp.Piece('p', False, 6, 6)
        guard_6 = chess_cpp.Piece('r', False, 7, 6)
        extra_2 = chess_cpp.Piece('n', False, 4, 6)
        boards.setPiece(0, my_king)
        boards.setPiece(0, guard_1)
        boards.setPiece(0, guard_2)
        boards.setPiece(0, guard_3)
        boards.setPiece(0, enemy_king)
        boards.setPiece(0, guard_4)
        boards.setPiece(0, guard_5)
        boards.setPiece(0, guard_6)
        boards.setPiece(0, extra_1)
        boards.setPiece(0, extra_2)
        batched_board = boards.to_tensor().cuda()
        bb_start = batched_board.clone()[0]
        our_ai_agent = DummyDQNMoveAgent(boards, bb_start, {"num_moves"})
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((BATCH_SIZE)).to(torch.bool).cuda()
        # Check the initial conditions
        assert our_ai_agent.previous_rewards is None
        assert our_ai_agent.previous_terminals is None
        assert our_ai_agent.opponent_rewards is None
        assert our_ai_agent.opponent_terminals is None
        # Make the first move, test the rewards in each case.
        current_state = batched_board.clone()
        desired_move = torch.Tensor([[1, 3, 2, 3]]).to(torch.int8).cuda()
        desired_promotion = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()  # irrelevant.
        our_ai_agent.store_move((desired_move, desired_promotion))
        (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        rewards = our_ai_agent.apply_all_rewards(game_over_tensor, (batched_board, opponent_move_layer))
        game_over = torch.any(game_over_tensor, dim=1)
        our_ai_agent.store_training_artifacts((current_state, move, promotion, rewards, game_over), opponent_move_layer)
        # Check how the rewards have changed.
        # Opponent now has 14 moves. 6 / sqrt(14 + 1) = 1.54919333848
        # 1.54919333848 + -1 = 0.54919333848
        assert our_ai_agent.previous_rewards is None
        assert our_ai_agent.previous_terminals is None
        assert torch.isclose(our_ai_agent.opponent_rewards, torch.Tensor([0.54919]))
        assert torch.isclose(rewards.cpu(), torch.Tensor([0.54919]))
        assert not our_ai_agent.opponent_terminals[0]
        assert our_ai_agent.experience_buffer.rewards_buffer.shape == (0,)
        assert our_ai_agent.experience_buffer.terminals_buffer.shape == (0,)
        # Make the second move, test the rewards in each case.
        current_state = batched_board.clone()
        desired_move = torch.Tensor([[3, 1, 2, 3]]).to(torch.int8).cuda()
        desired_promotion = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()  # irrelevant.
        our_ai_agent.store_move((desired_move, desired_promotion))
        (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        rewards = our_ai_agent.apply_all_rewards(game_over_tensor, (batched_board, opponent_move_layer))
        game_over = torch.any(game_over_tensor, dim=1)
        our_ai_agent.store_training_artifacts((current_state, move, promotion, rewards, game_over), opponent_move_layer)
        # Check how the rewards have changed.
        # Opponent now has 34 moves. 6 / sqrt(34 + 1) = 1.01418510567
        # 1.01418510567 + -1 = 0.01418510567
        # 0.54919333848 - 1.01418510567 = -0.46499176719
        assert torch.isclose(our_ai_agent.previous_rewards, torch.Tensor([-0.46499]))
        assert not our_ai_agent.previous_terminals[0]
        assert torch.isclose(our_ai_agent.opponent_rewards, torch.Tensor([0.014185]))
        assert torch.isclose(rewards.cpu(), torch.Tensor([0.014185]))
        assert not our_ai_agent.opponent_terminals[0]
        assert our_ai_agent.experience_buffer.rewards_buffer.shape == (0,)
        assert our_ai_agent.experience_buffer.terminals_buffer.shape == (0,)
        # Make the third move, test the rewards in each case.
        current_state = batched_board.clone()
        desired_move = torch.Tensor([[2, 3, 0, 3]]).to(torch.int8).cuda()
        desired_promotion = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()  # irrelevant.
        our_ai_agent.store_move((desired_move, desired_promotion))
        (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        rewards = our_ai_agent.apply_all_rewards(game_over_tensor, (batched_board, opponent_move_layer))
        game_over = torch.any(game_over_tensor, dim=1)
        our_ai_agent.store_training_artifacts((current_state, move, promotion, rewards, game_over), opponent_move_layer)
        # Check how the rewards have changed.
        # Opponent now has 17 moves. 6 / sqrt(17 + 1) = 1.41421356237
        # 1.41421356237 + -1 = 0.41421356237
        # 0.01418510567 - 1.41421356237 = -1.4000284567
        assert torch.isclose(our_ai_agent.previous_rewards, torch.Tensor([-1.40003]))
        assert not our_ai_agent.previous_terminals[0]
        assert torch.isclose(our_ai_agent.opponent_rewards, torch.Tensor([0.414214]))
        assert torch.isclose(rewards.cpu(), torch.Tensor([0.414214]))
        assert not our_ai_agent.opponent_terminals[0]
        assert torch.allclose(our_ai_agent.experience_buffer.rewards_buffer, torch.Tensor([-0.46499]))
        assert our_ai_agent.experience_buffer.terminals_buffer.shape == (1,)
        assert not our_ai_agent.experience_buffer.terminals_buffer[0]
    
    def test_checkmate(self):
        """
        Test how the rewards propagate in the checkmate case.
        """
        # Set up the board.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        my_king = chess_cpp.Piece('k', True, 0, 0)
        guard_1 = chess_cpp.Piece('p', True, 1, 0)
        guard_2 = chess_cpp.Piece('p', True, 1, 1)
        guard_3 = chess_cpp.Piece('r', True, 0, 7)
        enemy_king = chess_cpp.Piece('k', False, 7, 7)
        guard_4 = chess_cpp.Piece('p', False, 6, 7)
        guard_5 = chess_cpp.Piece('p', False, 6, 6)
        guard_6 = chess_cpp.Piece('r', False, 7, 6)
        extra_1 = chess_cpp.Piece('q', False, 5, 5)
        boards.setPiece(0, my_king)
        boards.setPiece(0, guard_1)
        boards.setPiece(0, guard_2)
        boards.setPiece(0, guard_3)
        boards.setPiece(0, enemy_king)
        boards.setPiece(0, guard_4)
        boards.setPiece(0, guard_5)
        boards.setPiece(0, guard_6)
        boards.setPiece(0, extra_1)
        batched_board = boards.to_tensor().cuda()
        bb_start = batched_board.clone()[0]
        our_ai_agent = DummyDQNMoveAgent(boards, bb_start, {"num_moves"})
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((BATCH_SIZE)).to(torch.bool).cuda()
        # Make the first move, test current rewards and terminals.
        current_state = batched_board.clone()
        desired_move = torch.Tensor([[0, 7, 1, 7]]).to(torch.int8).cuda()
        desired_promotion = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()  # irrelevant.
        our_ai_agent.store_move((desired_move, desired_promotion))
        (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        rewards = our_ai_agent.apply_all_rewards(game_over_tensor, (batched_board, opponent_move_layer))
        game_over = torch.any(game_over_tensor, dim=1)
        our_ai_agent.store_training_artifacts((current_state, move, promotion, rewards, game_over), opponent_move_layer)
        # Opponent now has 32 moves. 6 / sqrt(32 + 1) = 1.04446593573
        # 1.04446593573 + -1 = 0.04446593573
        assert torch.isclose(our_ai_agent.opponent_rewards, torch.Tensor([0.04446593573]))
        assert torch.isclose(rewards.cpu(), torch.Tensor([0.04446593573]))
        assert not our_ai_agent.opponent_terminals[0]
        # Make a checkmating move.
        current_state = batched_board.clone()
        desired_move = torch.Tensor([[2, 2, 7, 2]]).to(torch.int8).cuda()
        desired_promotion = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()  # irrelevant.
        our_ai_agent.store_move((desired_move, desired_promotion))
        (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        rewards = our_ai_agent.apply_all_rewards(game_over_tensor, (batched_board, opponent_move_layer))
        game_over = torch.any(game_over_tensor, dim=1)
        our_ai_agent.store_training_artifacts((current_state, move, promotion, rewards, game_over), opponent_move_layer)
        assert our_ai_agent.previous_rewards == torch.Tensor([-100])
        assert our_ai_agent.previous_terminals[0]
        assert our_ai_agent.opponent_rewards == 100
        assert rewards.cpu() == 100
        assert our_ai_agent.opponent_terminals[0]
    
    def test_firepower_function(self):
        # Set up the board.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        my_king = chess_cpp.Piece('k', True, 0, 0)
        guard_1 = chess_cpp.Piece('p', True, 1, 0)
        guard_2 = chess_cpp.Piece('p', True, 1, 1)
        guard_3 = chess_cpp.Piece('r', True, 0, 7)
        enemy_king = chess_cpp.Piece('k', False, 7, 7)
        guard_4 = chess_cpp.Piece('p', False, 6, 7)
        guard_5 = chess_cpp.Piece('p', False, 6, 6)
        guard_6 = chess_cpp.Piece('r', False, 7, 6)
        extra_1 = chess_cpp.Piece('n', False, 5, 5)
        boards.setPiece(0, my_king)
        boards.setPiece(0, guard_1)
        boards.setPiece(0, guard_2)
        boards.setPiece(0, guard_3)
        boards.setPiece(0, enemy_king)
        boards.setPiece(0, guard_4)
        boards.setPiece(0, guard_5)
        boards.setPiece(0, guard_6)
        boards.setPiece(0, extra_1)
        batched_board = boards.to_tensor().cuda()
        bb_start = batched_board.clone()[0]
        our_ai_agent = DummyDQNMoveAgent(boards, bb_start, {"firepower"})
        colour_list = torch.ones((BATCH_SIZE)).to(torch.bool).cuda()
        # Consider things the right way round.
        batched_board, _ = flip_board(batched_board, colour_list)
        # Remember, this function works in a flipped sense, so the "enemy knight" is our knight.
        # The difference of one knight should be a smallish reward.
        assert -0.2 <= our_ai_agent._firepower_score_fn(batched_board)[0] <= -0.1
