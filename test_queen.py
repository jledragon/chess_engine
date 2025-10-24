# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 01:15:38 2023

@author: jledragon
"""

# Run this with "python -m unittest test_queen.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestQueen(unittest.TestCase):
    '''
    Unit tests for queen.
    '''
    
    # All other behaviours should be tested in TestRook and TestBishop
    def testBasicMovesCentre(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('q', True, 3, 3)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 27
        assert to_board[3][2] == 1
        assert to_board[3][1] == 1
        assert to_board[3][0] == 1
        assert to_board[3][4] == 1
        assert to_board[3][5] == 1
        assert to_board[3][6] == 1
        assert to_board[3][7] == 1
        assert to_board[2][3] == 1
        assert to_board[1][3] == 1
        assert to_board[0][3] == 1
        assert to_board[4][3] == 1
        assert to_board[5][3] == 1
        assert to_board[6][3] == 1
        assert to_board[7][3] == 1
        assert to_board[2][2] == 1
        assert to_board[1][1] == 1
        assert to_board[0][0] == 1
        assert to_board[4][2] == 1
        assert to_board[5][1] == 1
        assert to_board[6][0] == 1
        assert to_board[4][4] == 1
        assert to_board[5][5] == 1
        assert to_board[6][6] == 1
        assert to_board[7][7] == 1
        assert to_board[2][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][6] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
