# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 23:14:27 2023

@author: jledragon
"""

# Run this with "python -m unittest test_movement.py"
import torch
import unittest
import chess_cpp


BATCH_SIZE = 1


class TestMovement(unittest.TestCase):
    '''
    Unit tests for movement
    '''
    
    def testSimplePawnMove(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 1, 3)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[1, 3, 3, 3]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][1][3] == 0
        assert batched_board[0][0][3][3] == 1
        assert batched_board[0][1][3][3] == 0
        assert batched_board[0][2][3][3] == 0
        assert batched_board[0][3][3][3] == 0
        assert batched_board[0][4][3][3] == 0
        assert batched_board[0][5][3][3] == 0
        assert batched_board[0][6][3][3] == 0
        assert batched_board[0][7][3][3] == 1
        assert dud_move_count[0][0] == 0
    
    def testStartingPawnMove(self):
        boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, 0)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[1, 0, 2, 0]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][1][0] == 0
        assert batched_board[0][0][2][0] == 1
        assert batched_board[0][1][2][0] == 0
        assert batched_board[0][2][2][0] == 0
        assert batched_board[0][3][2][0] == 0
        assert batched_board[0][4][2][0] == 0
        assert batched_board[0][5][2][0] == 0
        assert batched_board[0][6][2][0] == 0
        assert batched_board[0][7][2][0] == 1
        assert dud_move_count[0][0] == 0
    
    def testSimpleKnightTake(self):
        # Also tests taking.
        # Also tests resetting move count.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', True, 2, 3)
        enemy_queen = chess_cpp.Piece('p', False, 3, 5)
        boards.setPiece(0, knight)
        boards.setPiece(0, enemy_queen)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        dud_move_count[0][0] = 30
        move = torch.Tensor([[2, 3, 3, 5]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][2][3] == 0
        assert batched_board[0][0][3][5] == 0
        assert batched_board[0][1][3][5] == 1
        assert batched_board[0][2][3][5] == 0
        assert batched_board[0][3][3][5] == 0
        assert batched_board[0][4][3][5] == 0
        assert batched_board[0][5][3][5] == 0
        assert batched_board[0][6][3][5] == 0
        assert batched_board[0][7][3][5] == 1
        assert dud_move_count[0][0] == 0
    
    def testSimpleBishopMove(self):
        # Also tests this being the second move.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', True, 7, 7, False, True)
        boards.setPiece(0, bishop)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[7, 7, 0, 0]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][7][7] == 0
        assert batched_board[0][0][0][0] == 0
        assert batched_board[0][1][0][0] == 0
        assert batched_board[0][2][0][0] == 1
        assert batched_board[0][3][0][0] == 0
        assert batched_board[0][4][0][0] == 0
        assert batched_board[0][5][0][0] == 0
        assert batched_board[0][6][0][0] == 0
        assert batched_board[0][7][0][0] == 0
        assert dud_move_count[0][0] == 1
    
    def testSimpleRookMove(self):
        # Also tests this being neither the first nor the second move.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', True, 7, 7, False, False)
        boards.setPiece(0, rook)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[7, 7, 7, 0]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][7][7] == 0
        assert batched_board[0][0][7][0] == 0
        assert batched_board[0][1][7][0] == 0
        assert batched_board[0][2][7][0] == 0
        assert batched_board[0][3][7][0] == 1
        assert batched_board[0][4][7][0] == 0
        assert batched_board[0][5][7][0] == 0
        assert batched_board[0][6][7][0] == 0
        assert batched_board[0][7][7][0] == 0
        assert dud_move_count[0][0] == 1
    
    def testSimpleQueenMove(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        queen = chess_cpp.Piece('q', True, 3, 3)
        boards.setPiece(0, queen)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[3, 3, 3, 6]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][3][3] == 0
        assert batched_board[0][0][3][6] == 0
        assert batched_board[0][1][3][6] == 0
        assert batched_board[0][2][3][6] == 1
        assert batched_board[0][3][3][6] == 1
        assert batched_board[0][4][3][6] == 0
        assert batched_board[0][5][3][6] == 0
        assert batched_board[0][6][3][6] == 0
        assert batched_board[0][7][3][6] == 1
        assert dud_move_count[0][0] == 1
    
    def testSimpleKingMove(self):
        # Also tests high move count.
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        boards.setPiece(0, king)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        dud_move_count[0][0] = 30
        move = torch.Tensor([[3, 3, 4, 4]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][3][3] == 0
        assert batched_board[0][0][4][4] == 0
        assert batched_board[0][1][4][4] == 0
        assert batched_board[0][2][4][4] == 0
        assert batched_board[0][3][4][4] == 0
        assert batched_board[0][4][4][4] == 1
        assert batched_board[0][5][4][4] == 0
        assert batched_board[0][6][4][4] == 0
        assert batched_board[0][7][4][4] == 1
        assert dud_move_count[0][0] == 31
    
    def testEnPassantLeft(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 4, 0, False, False)
        enemy = chess_cpp.Piece('p', False, 4, 1, False, True)
        boards.setPiece(0, pawn)
        boards.setPiece(0, enemy)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[4, 0, 5, 1]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][4][0] == 0
            assert batched_board[0][i][4][1] == 0
        assert batched_board[0][0][5][1] == 1
        assert batched_board[0][1][5][1] == 0
        assert batched_board[0][2][5][1] == 0
        assert batched_board[0][3][5][1] == 0
        assert batched_board[0][4][5][1] == 0
        assert batched_board[0][5][5][1] == 0
        assert batched_board[0][6][5][1] == 0
        assert batched_board[0][7][5][1] == 0
        assert dud_move_count[0][0] == 0
    
    def testEnPassantRight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 4, 7, False, False)
        enemy = chess_cpp.Piece('p', False, 4, 6, False, True)
        boards.setPiece(0, pawn)
        boards.setPiece(0, enemy)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[4, 7, 5, 6]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][4][7] == 0
            assert batched_board[0][i][4][6] == 0
        assert batched_board[0][0][5][6] == 1
        assert batched_board[0][1][5][6] == 0
        assert batched_board[0][2][5][6] == 0
        assert batched_board[0][3][5][6] == 0
        assert batched_board[0][4][5][6] == 0
        assert batched_board[0][5][5][6] == 0
        assert batched_board[0][6][5][6] == 0
        assert batched_board[0][7][5][6] == 0
        assert dud_move_count[0][0] == 0
    
    def testQueenSideCastling(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 0)
        boards.setPiece(0, king)
        boards.setPiece(0, rook)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[0, 4, 0, 2]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][0][4] == 0
            assert batched_board[0][i][0][0] == 0
        assert batched_board[0][0][0][2] == 0
        assert batched_board[0][1][0][2] == 0
        assert batched_board[0][2][0][2] == 0
        assert batched_board[0][3][0][2] == 0
        assert batched_board[0][4][0][2] == 1
        assert batched_board[0][5][0][2] == 0
        assert batched_board[0][6][0][2] == 0
        assert batched_board[0][7][0][2] == 1

        assert batched_board[0][0][0][3] == 0
        assert batched_board[0][1][0][3] == 0
        assert batched_board[0][2][0][3] == 0
        assert batched_board[0][3][0][3] == 1
        assert batched_board[0][4][0][3] == 0
        assert batched_board[0][5][0][3] == 0
        assert batched_board[0][6][0][3] == 0
        assert batched_board[0][7][0][3] == 1
        assert dud_move_count[0][0] == 1
    
    def testKingSideCastling(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 0, 4)
        rook = chess_cpp.Piece('r', True, 0, 7)
        boards.setPiece(0, king)
        boards.setPiece(0, rook)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[0, 4, 0, 6]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_nothing, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][0][4] == 0
            assert batched_board[0][i][0][7] == 0
        assert batched_board[0][0][0][6] == 0
        assert batched_board[0][1][0][6] == 0
        assert batched_board[0][2][0][6] == 0
        assert batched_board[0][3][0][6] == 0
        assert batched_board[0][4][0][6] == 1
        assert batched_board[0][5][0][6] == 0
        assert batched_board[0][6][0][6] == 0
        assert batched_board[0][7][0][6] == 1

        assert batched_board[0][0][0][5] == 0
        assert batched_board[0][1][0][5] == 0
        assert batched_board[0][2][0][5] == 0
        assert batched_board[0][3][0][5] == 1
        assert batched_board[0][4][0][5] == 0
        assert batched_board[0][5][0][5] == 0
        assert batched_board[0][6][0][5] == 0
        assert batched_board[0][7][0][5] == 1
        assert dud_move_count[0][0] == 1
    
    def testPawnPromotionKnight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 6, 0, False, False)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_knight = torch.Tensor([[1, 0, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[6, 0, 7, 0]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_knight, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][6][0] == 0
        assert batched_board[0][0][7][0] == 0
        assert batched_board[0][1][7][0] == 1
        assert batched_board[0][2][7][0] == 0
        assert batched_board[0][3][7][0] == 0
        assert batched_board[0][4][7][0] == 0
        assert batched_board[0][5][7][0] == 0
        assert batched_board[0][6][7][0] == 0
        assert batched_board[0][7][7][0] == 0
        assert dud_move_count[0][0] == 0
        
    
    def testPawnPromotionBishop(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 6, 0, False, False)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_knight = torch.Tensor([[0, 1, 0, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[6, 0, 7, 0]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_knight, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][6][0] == 0
        assert batched_board[0][0][7][0] == 0
        assert batched_board[0][1][7][0] == 0
        assert batched_board[0][2][7][0] == 1
        assert batched_board[0][3][7][0] == 0
        assert batched_board[0][4][7][0] == 0
        assert batched_board[0][5][7][0] == 0
        assert batched_board[0][6][7][0] == 0
        assert batched_board[0][7][7][0] == 0
        assert dud_move_count[0][0] == 0
    
    def testPawnPromotionRook(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 6, 0, False, False)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_knight = torch.Tensor([[0, 0, 1, 0]]).to(torch.int8).cuda()
        move = torch.Tensor([[6, 0, 7, 0]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_knight, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][6][0] == 0
        assert batched_board[0][0][7][0] == 0
        assert batched_board[0][1][7][0] == 0
        assert batched_board[0][2][7][0] == 0
        assert batched_board[0][3][7][0] == 1
        assert batched_board[0][4][7][0] == 0
        assert batched_board[0][5][7][0] == 0
        assert batched_board[0][6][7][0] == 0
        assert batched_board[0][7][7][0] == 0
        assert dud_move_count[0][0] == 0
    
    def testPawnPromotionQueen(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', True, 6, 0, False, False)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list().cuda()
        promotion_knight = torch.Tensor([[0, 0, 0, 1]]).to(torch.int8).cuda()
        move = torch.Tensor([[6, 0, 7, 0]]).to(torch.int8).cuda()
        chess_cpp.enact_moves(batched_board, move, promotion_knight, dud_move_count)
        for i in range(0, 8):
            assert batched_board[0][i][6][0] == 0
        assert batched_board[0][0][7][0] == 0
        assert batched_board[0][1][7][0] == 0
        assert batched_board[0][2][7][0] == 1
        assert batched_board[0][3][7][0] == 1
        assert batched_board[0][4][7][0] == 0
        assert batched_board[0][5][7][0] == 0
        assert batched_board[0][6][7][0] == 0
        assert batched_board[0][7][7][0] == 0
        assert dud_move_count[0][0] == 0
