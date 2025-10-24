# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:15:38 2023

@author: jledragon
"""

# Run this with "python -m unittest test_bishop.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestBishop(unittest.TestCase):
    '''
    Unit tests for bishop.
    '''

    def testBasicMovesCentreLight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', True, 3, 3)
        boards.setPiece(1, bishop)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 13
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
    
    def testBasicMovesCentreDark(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', True, 4, 3)
        boards.setPiece(1, bishop)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][4*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 13
        assert to_board[3][2] == 1
        assert to_board[2][1] == 1
        assert to_board[1][0] == 1
        assert to_board[5][4] == 1
        assert to_board[6][5] == 1
        assert to_board[7][6] == 1
        assert to_board[3][4] == 1
        assert to_board[2][5] == 1
        assert to_board[1][6] == 1
        assert to_board[0][7] == 1
        assert to_board[5][2] == 1
        assert to_board[6][1] == 1
        assert to_board[7][0] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testEnemiesInTheWay(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', True, 3, 3)
        enemy1 = chess_cpp.Piece('b', False, 2, 2)
        enemy2 = chess_cpp.Piece('b', False, 4, 4)
        enemy3 = chess_cpp.Piece('b', False, 4, 2)
        enemy4 = chess_cpp.Piece('b', False, 2, 4)
        boards.setPiece(1, bishop)
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
        assert to_board[2][2] == 1
        assert to_board[4][4] == 1
        assert to_board[2][4] == 1
        assert to_board[4][2] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testFriendsInTheWay(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', True, 3, 3)
        friend1 = chess_cpp.Piece('b', True, 2, 2)
        friend2 = chess_cpp.Piece('b', True, 4, 4)
        friend3 = chess_cpp.Piece('b', True, 4, 2)
        friend4 = chess_cpp.Piece('b', True, 2, 4)
        boards.setPiece(1, bishop)
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
