# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:32:36 2023

@author: jledragon
"""

# Run this with "python -m unittest test_check.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestKing(unittest.TestCase):
    '''
    Unit tests for check cases.
    '''
    
    def test_simple_rook_cases(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[3][3] == 0
        assert torch.sum(relevant_board) == 14

        assert relevant_board[3][2] == 1
        assert relevant_board[3][1] == 1
        assert relevant_board[3][0] == 1
        assert relevant_board[3][4] == 1
        assert relevant_board[3][5] == 1
        assert relevant_board[3][6] == 1
        assert relevant_board[3][7] == 1
        
        assert relevant_board[2][3] == 1
        assert relevant_board[1][3] == 1
        assert relevant_board[0][3] == 1
        assert relevant_board[4][3] == 1
        assert relevant_board[5][3] == 1
        assert relevant_board[6][3] == 1
        assert relevant_board[7][3] == 1

        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    def test_rook_cases_pieces_in_the_way(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        my_piece_1 = chess_cpp.Piece('r', True, 3, 2)
        my_piece_2 = chess_cpp.Piece('k', True, 3, 6)
        enemy_piece_1 = chess_cpp.Piece('p', False, 0, 3)
        my_piece_3 = chess_cpp.Piece('n', True, 5, 3)
        boards.setPiece(1, rook)
        boards.setPiece(1, my_piece_1)
        boards.setPiece(1, my_piece_2)
        boards.setPiece(1, my_piece_3)
        # Recall, we ask "if you were a king, where would you be in check?"
        # Therefore this enemy piece should register as a 1.
        boards.setPiece(1, enemy_piece_1)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[3][3] == 0
        assert torch.sum(relevant_board) == 9

        assert relevant_board[3][2] == 1
        assert relevant_board[3][4] == 1
        assert relevant_board[3][5] == 1
        assert relevant_board[3][6] == 1
        
        assert relevant_board[2][3] == 1
        assert relevant_board[1][3] == 1
        assert relevant_board[0][3] == 1
        assert relevant_board[4][3] == 1
        assert relevant_board[5][3] == 1

        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    def test_simple_bishop_cases(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', False, 3, 3)
        boards.setPiece(1, bishop)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[3][3] == 0
        assert torch.sum(relevant_board) == 13

        assert relevant_board[2][2] == 1
        assert relevant_board[1][1] == 1
        assert relevant_board[0][0] == 1
        assert relevant_board[4][4] == 1
        assert relevant_board[5][5] == 1
        assert relevant_board[6][6] == 1
        assert relevant_board[7][7] == 1
        
        assert relevant_board[2][4] == 1
        assert relevant_board[1][5] == 1
        assert relevant_board[0][6] == 1
        assert relevant_board[4][2] == 1
        assert relevant_board[5][1] == 1
        assert relevant_board[6][0] == 1

        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    def test_bishop_cases_pieces_in_the_way(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', False, 3, 3)
        enemy_piece_1 = chess_cpp.Piece('p', False, 0, 0)
        my_piece_1 = chess_cpp.Piece('b', True, 2, 4)
        my_piece_2 = chess_cpp.Piece('k', True, 6, 6)
        my_piece_3 = chess_cpp.Piece('n', True, 5, 1)
        boards.setPiece(1, bishop)
        boards.setPiece(1, my_piece_1)
        boards.setPiece(1, my_piece_2)
        boards.setPiece(1, my_piece_3)
        # Recall, we ask "if you were a king, where would you be in check?"
        # Therefore this enemy piece should register as a 1.
        boards.setPiece(1, enemy_piece_1)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[3][3] == 0
        assert torch.sum(relevant_board) == 9

        assert relevant_board[2][2] == 1
        assert relevant_board[1][1] == 1
        assert relevant_board[0][0] == 1
        assert relevant_board[4][4] == 1
        assert relevant_board[5][5] == 1
        assert relevant_board[6][6] == 1
        
        assert relevant_board[2][4] == 1
        assert relevant_board[4][2] == 1
        assert relevant_board[5][1] == 1
        
        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    def test_queen_cases(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        queen = chess_cpp.Piece('q', False, 4, 4)
        boards.setPiece(1, queen)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[4][4] == 0
        assert torch.sum(relevant_board) == 27
        
        assert relevant_board[3][3] == 1
        assert relevant_board[2][2] == 1
        assert relevant_board[1][1] == 1
        assert relevant_board[0][0] == 1
        assert relevant_board[5][5] == 1
        assert relevant_board[6][6] == 1
        assert relevant_board[7][7] == 1
        
        assert relevant_board[3][5] == 1
        assert relevant_board[2][6] == 1
        assert relevant_board[1][7] == 1
        assert relevant_board[5][3] == 1
        assert relevant_board[6][2] == 1
        assert relevant_board[7][1] == 1
        
        assert relevant_board[4][3] == 1
        assert relevant_board[4][2] == 1
        assert relevant_board[4][1] == 1
        assert relevant_board[4][0] == 1
        assert relevant_board[4][5] == 1
        assert relevant_board[4][6] == 1
        assert relevant_board[4][7] == 1
        
        assert relevant_board[3][4] == 1
        assert relevant_board[2][4] == 1
        assert relevant_board[1][4] == 1
        assert relevant_board[0][4] == 1
        assert relevant_board[5][4] == 1
        assert relevant_board[6][4] == 1
        assert relevant_board[7][4] == 1
        
        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    def test_knight_cases(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        knight = chess_cpp.Piece('n', False, 3, 3)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[3][3] == 0
        assert torch.sum(relevant_board) == 8
        
        assert relevant_board[1][2] == 1
        assert relevant_board[2][1] == 1
        assert relevant_board[4][1] == 1
        assert relevant_board[2][5] == 1
        assert relevant_board[1][4] == 1
        assert relevant_board[2][5] == 1
        assert relevant_board[4][5] == 1
        assert relevant_board[5][4] == 1
        
        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    def test_pawn_cases(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        pawn = chess_cpp.Piece('p', False, 3, 3)
        boards.setPiece(1, pawn)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[3][3] == 0
        assert torch.sum(relevant_board) == 2
        
        assert relevant_board[2][2] == 1
        assert relevant_board[2][4] == 1
        
        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    def test_king_cases(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', False, 3, 3)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        squares_in_check = chess_cpp.get_squares_in_check_for_player(batched_board)
        relevant_board = squares_in_check[1].reshape((8, 8))
        assert relevant_board[3][3] == 0
        assert torch.sum(relevant_board) == 8
        
        assert relevant_board[2][2] == 1
        assert relevant_board[2][3] == 1
        assert relevant_board[2][4] == 1
        assert relevant_board[3][2] == 1
        assert relevant_board[3][4] == 1
        assert relevant_board[4][2] == 1
        assert relevant_board[4][3] == 1
        assert relevant_board[4][4] == 1
        
        irrelevant_1 = squares_in_check[0]
        irrelevant_2 = squares_in_check[2]
        assert torch.sum(irrelevant_1) == 0
        assert torch.sum(irrelevant_2) == 0
    
    # TODO - add practical cases once checkmate is implemented.
