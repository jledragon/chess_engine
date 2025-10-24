# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 01:20:58 2023

@author: jledragon
"""

# Run this with "python -m unittest test_king.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestKing(unittest.TestCase):
    '''
    Unit tests for king.
    '''
    
    def testBasicMovesCentre(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 8
        assert to_board[2][2] == 1
        assert to_board[2][3] == 1
        assert to_board[2][4] == 1
        assert to_board[3][2] == 1
        assert to_board[3][4] == 1
        assert to_board[4][2] == 1
        assert to_board[4][3] == 1
        assert to_board[4][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicMovesBottomLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 0)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 0].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[0][1] == 1
        assert to_board[1][0] == 1
        assert to_board[1][1] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicMovesTopLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 7)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 7].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[0][6] == 1
        assert to_board[1][6] == 1
        assert to_board[1][7] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicMovesTopRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 7, 7)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][7*8 + 7].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[6][6] == 1
        assert to_board[6][7] == 1
        assert to_board[7][6] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicMovesBottomRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 7, 0)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][7*8 + 0].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[6][0] == 1
        assert to_board[6][1] == 1
        assert to_board[7][1] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicMovesEnemies(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        enemy = chess_cpp.Piece('p', False, 3, 2)
        boards.setPiece(1, king)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 7
        assert to_board[2][2] == 1
        assert to_board[2][4] == 1
        assert to_board[3][2] == 1
        assert to_board[3][4] == 1
        assert to_board[4][2] == 1
        assert to_board[4][3] == 1
        assert to_board[4][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicMovesFriends(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        friend1 = chess_cpp.Piece('p', True, 2, 2)
        friend2 = chess_cpp.Piece('p', True, 3, 4)
        boards.setPiece(1, king)
        boards.setPiece(1, friend1)
        boards.setPiece(1, friend2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 6
        assert to_board[2][3] == 1
        assert to_board[2][4] == 1
        assert to_board[3][2] == 1
        assert to_board[4][2] == 1
        assert to_board[4][3] == 1
        assert to_board[4][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testQueenCastlingBlocked_1(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 0)
        queen = chess_cpp.Piece('q', True, 0, 3)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 4
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testQueenCastlingBlocked_2(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 0)
        enemy_knight = chess_cpp.Piece('n', False, 0, 1)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 4
        assert to_board[0][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testKingCastlingBlocked_1(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7)
        queen = chess_cpp.Piece('q', True, 0, 5)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 4
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testKingCastlingBlocked_2(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7)
        enemy_knight = chess_cpp.Piece('n', False, 0, 6)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 4
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicQueensideCastling(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 0)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 6
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert to_board[0][2] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBasicKingsideCastling(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 6
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert to_board[0][6] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingKingNotFirstMove(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4, False, False)
        rook = chess_cpp.Piece('r', True, 0, 7)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 5
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingRookNotFirstMove(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7, False, False)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 5
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingQueensCheck_1(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 0)
        enemy_piece = chess_cpp.Piece('q', False, 5, 4)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_piece)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 4
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingQueensCheck_2(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 0)
        enemy_piece = chess_cpp.Piece('q', False, 5, 3)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_piece)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingQueensCheck_3(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 0)
        enemy_piece = chess_cpp.Piece('q', False, 5, 2)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_piece)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 5
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingKingsCheck_1(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7)
        enemy_piece = chess_cpp.Piece('q', False, 5, 4)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_piece)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 4
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingKingsCheck_2(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7)
        enemy_piece = chess_cpp.Piece('q', False, 5, 5)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_piece)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCastlingKingsCheck_3(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7)
        enemy_piece = chess_cpp.Piece('q', False, 5, 6)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, enemy_piece)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 5
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCannotMoveIntoCheck(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        enemy_1 = chess_cpp.Piece('r', False, 2, 0)
        enemy_2 = chess_cpp.Piece('b', False, 5, 3)
        enemy_3 = chess_cpp.Piece('n', False, 5, 6)
        boards.setPiece(1, king)
        boards.setPiece(1, enemy_1)
        boards.setPiece(1, enemy_2)
        boards.setPiece(1, enemy_3)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[3][2] == 1
        assert to_board[3][4] == 1
        assert to_board[4][3] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testCannotMoveBackwardsIntoCheck(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        enemy = chess_cpp.Piece('q', False, 0, 3)
        boards.setPiece(1, king)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][3*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 6
        assert to_board[2][2] == 1
        assert to_board[3][2] == 1
        assert to_board[4][2] == 1
        assert to_board[2][4] == 1
        assert to_board[3][4] == 1
        assert to_board[4][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def test_cannot_castle_with_queen_1(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        queen = chess_cpp.Piece('q', True, 0, 7)
        boards.setPiece(1, king)
        boards.setPiece(1, queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 5
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def test_cannot_castle_with_queen_2(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        queen = chess_cpp.Piece('q', True, 0, 0)
        boards.setPiece(1, king)
        boards.setPiece(1, queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][0*8 + 4].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 5
        assert to_board[0][3] == 1
        assert to_board[1][3] == 1
        assert to_board[1][4] == 1
        assert to_board[1][5] == 1
        assert to_board[0][5] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
