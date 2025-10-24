# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:21:06 2023

@author: jledragon
"""

# Run this with "python -m unittest test_pawn.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestPawn(unittest.TestCase):
    '''
    Unit tests for pawns.
    '''
    
    def testBasicMoves(self):
        # Note - we do not test black here due to the fact that we code the board
        # to give us moves for "us". We will however, need to test that the board
        # can be flipped to black's perspective.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 3)
        boards.setPiece(1, pawn)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[2][3] == 1
        assert to_board[3][3] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testSingleMoveNotATakingMove(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 3)
        enemy_queen = chess_cpp.Piece('q', False, 2, 3)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy_queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 0
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testDoubleMoveNotATakingMove(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 3)
        enemy_queen = chess_cpp.Piece('q', False, 3, 3)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy_queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 1
        assert to_board[2][3] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testBoundary(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 7, 7)
        boards.setPiece(1, pawn)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][7*8 + 7].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 0
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testNonFirstMove(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 2, 2, False, False)
        boards.setPiece(1, pawn)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][2*8 + 2].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 1
        assert to_board[3][2] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testTakingLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 3)
        enemy = chess_cpp.Piece('p', False, 2, 2)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[2][3] == 1
        assert to_board[3][3] == 1
        assert to_board[2][2] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testTakingRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 3)
        enemy = chess_cpp.Piece('p', False, 2, 4)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 3
        assert to_board[2][3] == 1
        assert to_board[3][3] == 1
        assert to_board[2][4] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testEdgeTakingLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 0)
        enemy = chess_cpp.Piece('p', False, 1, 7)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 0].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[2][0] == 1
        assert to_board[3][0] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testEdgeTakingRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 7)
        enemy = chess_cpp.Piece('p', False, 3, 0)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 7].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[2][7] == 1
        assert to_board[3][7] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testTakingOwnColour(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 3)
        not_enemy = chess_cpp.Piece('p', True, 2, 2)
        boards.setPiece(1, pawn)
        boards.setPiece(1, not_enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][1*8 + 3].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[2][3] == 1
        assert to_board[3][3] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testEnPassantLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 4, 0, False, False)
        enemy = chess_cpp.Piece('p', False, 4, 1, False, True)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][4*8 + 0].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[5][0] == 1
        assert to_board[5][1] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testEnPassantRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 4, 7, False, False)
        enemy = chess_cpp.Piece('p', False, 4, 6, False, True)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][4*8 + 7].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 2
        assert to_board[5][7] == 1
        assert to_board[5][6] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
    
    def testEnPassantExpires(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 4, 0, False, False)
        enemy_pawn = chess_cpp.Piece('p', False, 4, 1, False, True)
        enemy_bishop = chess_cpp.Piece('b', False, 0, 0)
        boards.setPiece(1, pawn)
        boards.setPiece(1, enemy_pawn)
        boards.setPiece(1, enemy_bishop)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[0, 0, 7, 7]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        to_board = move_layer[1][4*8 + 0].reshape((8, 8))
        to_left = move_layer[0]
        to_right = move_layer[2]
        assert torch.sum(to_board) == 1
        assert to_board[5][0] == 1
        assert torch.sum(to_left) == 0
        assert torch.sum(to_right) == 0
