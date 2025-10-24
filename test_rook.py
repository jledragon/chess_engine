# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:40:32 2023

@author: jledragon
"""

# Run this with "python -m unittest test_rook.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestRook(unittest.TestCase):
    '''
    Unit tests for rook.
    '''
    
    def testBasicMovesCentre(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', True, 3, 3)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 14
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
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testEnemiesInTheWay(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', True, 3, 3)
        enemy1 = chess_cpp.Piece('r', False, 4, 3)
        enemy2 = chess_cpp.Piece('r', False, 3, 4)
        enemy3 = chess_cpp.Piece('r', False, 2, 3)
        enemy4 = chess_cpp.Piece('r', False, 3, 2)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy1)
        boards.setPiece(1, enemy2)
        boards.setPiece(1, enemy3)
        boards.setPiece(1, enemy4)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 4
        assert to_board[3][2] == 1
        assert to_board[2][3] == 1
        assert to_board[4][3] == 1
        assert to_board[3][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testFriendsInTheWay(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', True, 3, 3)
        friend1 = chess_cpp.Piece('r', True, 4, 3)
        friend2 = chess_cpp.Piece('r', True, 3, 4)
        friend3 = chess_cpp.Piece('r', True, 2, 3)
        friend4 = chess_cpp.Piece('r', True, 3, 2)
        boards.setPiece(1, rook)
        boards.setPiece(1, friend1)
        boards.setPiece(1, friend2)
        boards.setPiece(1, friend3)
        boards.setPiece(1, friend4)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 0
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
