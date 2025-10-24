# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:46:40 2023

@author: jledragon
"""

# Run this with "python -m unittest test_knight.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestKnight(unittest.TestCase):
    '''
    Unit tests for knights.
    '''
    
    def testBasicMovesCentre(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 3, 3)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 8
        assert to_board[1][2] == 1
        assert to_board[1][4] == 1
        assert to_board[2][1] == 1
        assert to_board[2][5] == 1
        assert to_board[4][1] == 1
        assert to_board[4][5] == 1
        assert to_board[5][2] == 1
        assert to_board[5][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testTakingCentre(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 3, 3)
        enemy1 = chess_cpp.Piece('b', False, 1, 2)
        enemy2 = chess_cpp.Piece('q', False, 4, 1)
        boards.setPiece(1, knight)
        boards.setPiece(1, enemy1)
        boards.setPiece(1, enemy2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 8
        assert to_board[1][2] == 1
        assert to_board[1][4] == 1
        assert to_board[2][1] == 1
        assert to_board[2][5] == 1
        assert to_board[4][1] == 1
        assert to_board[4][5] == 1
        assert to_board[5][2] == 1
        assert to_board[5][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testFriends(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 3, 3)
        friend1 = chess_cpp.Piece('b', True, 1, 2)
        friend2 = chess_cpp.Piece('q', True, 4, 1)
        boards.setPiece(1, knight)
        boards.setPiece(1, friend1)
        boards.setPiece(1, friend2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 6
        assert to_board[1][4] == 1
        assert to_board[2][1] == 1
        assert to_board[2][5] == 1
        assert to_board[4][5] == 1
        assert to_board[5][2] == 1
        assert to_board[5][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBottomLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 0, 0)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 0].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[2][1] == 1
        assert to_board[1][2] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testTopLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 0, 7)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 7].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[1][5] == 1
        assert to_board[2][6] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testTopRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 7, 0)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][7*8 + 0].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[5][1] == 1
        assert to_board[6][2] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBottomRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 7, 7)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][7*8 + 7].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[5][6] == 1
        assert to_board[6][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
