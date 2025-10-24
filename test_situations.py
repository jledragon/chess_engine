# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:38:20 2023

@author: jledragon
"""

# Run this with "python -m unittest test_situations.py"
import torch
import unittest
import chess_cpp
from chess_py_utils import is_game_over, get_repetition_status
#import time

BATCH_SIZE = 3


class TestSituations(unittest.TestCase):
    '''
    Unit tests for entire situations.
    '''
    
    def test_starting_position(self):
        boards = chess_cpp.BatchedBoard(True, 1, 0)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        move_layer = move_layer.reshape((1, 64, 8, 8))
        all_moves = torch.sum(move_layer, (0, 1))
        assert torch.sum(all_moves) == 20
        assert all_moves[2][0] == 2
        assert all_moves[3][0] == 1
        assert all_moves[2][1] == 1
        assert all_moves[3][1] == 1
        assert all_moves[2][2] == 2
        assert all_moves[3][2] == 1
        assert all_moves[2][3] == 1
        assert all_moves[3][3] == 1
        assert all_moves[2][4] == 1
        assert all_moves[3][4] == 1
        assert all_moves[2][5] == 2
        assert all_moves[3][5] == 1
        assert all_moves[2][6] == 1
        assert all_moves[3][6] == 1
        assert all_moves[2][7] == 2
        assert all_moves[3][7] == 1
    
    def test_legacy_case_1(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 0)
        pawn1 = chess_cpp.Piece('p', True, 1, 1)
        pawn2 = chess_cpp.Piece('p', True, 3, 1)
        pawn3 = chess_cpp.Piece('p', True, 4, 1)
        queen = chess_cpp.Piece('q', True, 4, 0)
        bishop = chess_cpp.Piece('b', True, 2, 0)
        enemyQueen = chess_cpp.Piece('q', False, 0, 3)
        boards.setPiece(1, king)
        boards.setPiece(1, pawn1)
        boards.setPiece(1, pawn2)
        boards.setPiece(1, pawn3)
        boards.setPiece(1, queen)
        boards.setPiece(1, bishop)
        boards.setPiece(1, enemyQueen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        assert torch.sum(move_layer[1]) == 1
        to_board = move_layer[1][1*8 + 1].reshape((8, 8))
        assert(to_board[2][1] == 1)
    
    def test_legacy_case_2(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king = chess_cpp.Piece('k', True, 0, 7)
        rook1 = chess_cpp.Piece('r', False, 1, 5)
        rook2 = chess_cpp.Piece('r', False, 2, 4)
        rook3 = chess_cpp.Piece('r', False, 6, 6)
        enemy_king = chess_cpp.Piece('k', False, 4, 0)
        boards.setPiece(1, king)
        boards.setPiece(1, rook1)
        boards.setPiece(1, rook2)
        boards.setPiece(1, rook3)
        boards.setPiece(1, enemy_king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[1]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == True  # Stalemate
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def test_legacy_case_3(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king = chess_cpp.Piece('k', True, 2, 0)
        queen = chess_cpp.Piece('q', False, 2, 1)
        rook = chess_cpp.Piece('r', False, 1, 1)
        enemy_king = chess_cpp.Piece('k', False, 7, 7)
        boards.setPiece(1, king)
        boards.setPiece(1, queen)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[1]
        assert game_over_tensor[0] == True # Checkmate
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def test_legacy_case_4(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king = chess_cpp.Piece('k', True, 1, 5)
        rook = chess_cpp.Piece('r', False, 3, 6)
        bishop = chess_cpp.Piece('b', False, 4, 5)
        pawn = chess_cpp.Piece('p', False, 6, 5)
        enemy_king = chess_cpp.Piece('k', False, 3, 7)
        boards.setPiece(1, king)
        boards.setPiece(1, bishop)
        boards.setPiece(1, rook)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy_king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[1]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    """
    def test_speed(self):
        # This is a bit more of an integration test.
        # Decide what to do with it.
        # Currently, roughly 60 x batched (256) moves per second.
        boards = chess_cpp.BatchedBoard(True, 256, 0)
        batched_board = boards.to_tensor().cuda()
        now = time.time()
        for i in range(0, 60):
            move_layer = chess_cpp.get_moves_for_player(batched_board)
        elapsed = time.time() - now
        print(elapsed)
    """
