# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:03:02 2023

@author: jledragon
"""

# Run this with "python -m unittest test_endgames.py"
import torch
import unittest
import chess_cpp
from chess_py_utils import is_game_over, get_repetition_status, reset_move_counts


BATCH_SIZE = 1


class TestEndgames(unittest.TestCase):
    '''
    Unit tests for endgame cases.
    '''
    
    def testSimpleCheckmate(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        enemyQueen1 = chess_cpp.Piece('q', False, 3, 5)
        enemyQueen2 = chess_cpp.Piece('q', False, 3, 3)
        enemyRook = chess_cpp.Piece('r', False, 3, 4)
        king = chess_cpp.Piece('k', True, 5, 4)
        queen = chess_cpp.Piece('q', True, 0, 0)
        knight = chess_cpp.Piece('n', True, 1, 0)
        bishop = chess_cpp.Piece('b', True, 2, 0)
        rook = chess_cpp.Piece('r', True, 3, 0)
        pawn = chess_cpp.Piece('p', True, 4, 0)
        boards.setPiece(0, enemyQueen1)
        boards.setPiece(0, enemyQueen2)
        boards.setPiece(0, enemyRook)
        boards.setPiece(0, king)
        boards.setPiece(0, queen)
        boards.setPiece(0, knight)
        boards.setPiece(0, bishop)
        boards.setPiece(0, rook)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == True # Checkmate
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testNotCheckmatebutCheck(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        enemyQueen = chess_cpp.Piece('q', False, 3, 5)
        enemyRook = chess_cpp.Piece('r', False, 3, 4)
        king = chess_cpp.Piece('k', True, 5, 4)
        queen = chess_cpp.Piece('q', True, 0, 0)
        knight = chess_cpp.Piece('n', True, 1, 0)
        bishop = chess_cpp.Piece('b', True, 2, 0)
        rook = chess_cpp.Piece('r', True, 3, 0)
        pawn = chess_cpp.Piece('p', True, 4, 0)
        boards.setPiece(0, enemyQueen)
        boards.setPiece(0, enemyRook)
        boards.setPiece(0, king)
        boards.setPiece(0, queen)
        boards.setPiece(0, knight)
        boards.setPiece(0, bishop)
        boards.setPiece(0, rook)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
        
    def testSimpleStalemate(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        enemyKnight1 = chess_cpp.Piece('n', False, 6, 7)
        enemyKnight2 = chess_cpp.Piece('n', False, 6, 6)
        enemyKnight3 = chess_cpp.Piece('n', False, 7, 6)
        enemyQueen = chess_cpp.Piece('q', False, 3, 7)
        enemyBishop = chess_cpp.Piece('b', False, 3, 3)
        enemyRook = chess_cpp.Piece('r', False, 7, 3)
        king = chess_cpp.Piece('k', True, 7, 7)
        boards.setPiece(0, enemyKnight1)
        boards.setPiece(0, enemyKnight2)
        boards.setPiece(0, enemyKnight3)
        boards.setPiece(0, enemyQueen)
        boards.setPiece(0, enemyBishop)
        boards.setPiece(0, enemyRook)
        boards.setPiece(0, king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == True # Stalemate
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testSimpleFiftyMoveRule(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        dud_move_count[0][0] = 50
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        queen = chess_cpp.Piece('q', False, 0, 5)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)
        game_over_0 = game_over_tensor[0]
        assert game_over_0[0] == False
        assert game_over_0[1] == False
        assert game_over_0[2] == False
        assert game_over_0[3] == False
        assert game_over_0[4] == True # 50 move rule
        # Also tests resetting moves
        assert dud_move_count[0] == 50
        game_over = torch.any(game_over_tensor, dim=1)
        reset_move_counts(dud_move_count, game_over)
        assert dud_move_count[0] == 0
    
    def testNotStalemateBecauseCheck(self):
        # Actually a checkmate. Don't know why I wrote this test originally...
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        enemyQueen1 = chess_cpp.Piece('q', False, 3, 5)
        enemyQueen2 = chess_cpp.Piece('q', False, 3, 3)
        enemyRook = chess_cpp.Piece('r', False, 3, 4)
        king = chess_cpp.Piece('k', True, 5, 4)
        queen = chess_cpp.Piece('q', True, 0, 0)
        knight = chess_cpp.Piece('n', True, 1, 0)
        bishop = chess_cpp.Piece('b', True, 2, 0)
        rook = chess_cpp.Piece('r', True, 3, 0)
        pawn = chess_cpp.Piece('p', True, 4, 0)
        boards.setPiece(0, enemyQueen1)
        boards.setPiece(0, enemyQueen2)
        boards.setPiece(0, enemyRook)
        boards.setPiece(0, king)
        boards.setPiece(0, queen)
        boards.setPiece(0, knight)
        boards.setPiece(0, bishop)
        boards.setPiece(0, rook)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == True # Checkmate
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testNotStalemate(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        enemyKnight1 = chess_cpp.Piece('n', False, 6, 7)
        enemyKnight2 = chess_cpp.Piece('n', False, 6, 6)
        enemyKnight3 = chess_cpp.Piece('n', False, 7, 6)
        king = chess_cpp.Piece('k', True, 7, 7)
        boards.setPiece(0, enemyKnight1)
        boards.setPiece(0, enemyKnight2)
        boards.setPiece(0, enemyKnight3)
        boards.setPiece(0, king)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testTwoKings(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == True # Insufficient material
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testOneKnight(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        knight = chess_cpp.Piece('n', True, 0, 5)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, knight)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == True # Insufficient material
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
        
    def testOneBishop(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        bishop = chess_cpp.Piece('b', False, 0, 5)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, bishop)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == True # Insufficient material
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testOneQueen(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        queen = chess_cpp.Piece('q', False, 0, 5)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, queen)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testOnePawn(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        pawn = chess_cpp.Piece('p', False, 1, 5)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, pawn)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testTwoLightBishops(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        bishop1 = chess_cpp.Piece('b', True, 0, 0)
        bishop2 = chess_cpp.Piece('b', True, 0, 2)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, bishop1)
        boards.setPiece(0, bishop2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == True # Insufficient material
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testTwoDarkBishops(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        bishop1 = chess_cpp.Piece('b', True, 0, 1)
        bishop2 = chess_cpp.Piece('b', False, 0, 3)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, bishop1)
        boards.setPiece(0, bishop2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == True # Insufficient material
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testLightAndDarkBishop(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        bishop1 = chess_cpp.Piece('b', True, 0, 1)
        bishop2 = chess_cpp.Piece('b', False, 0, 2)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, bishop1)
        boards.setPiece(0, bishop2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == False
        assert game_over_tensor[4] == False
    
    def testTwoKnights(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        knight1 = chess_cpp.Piece('n', True, 0, 1)
        knight2 = chess_cpp.Piece('n', False, 0, 2)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, knight1)
        boards.setPiece(0, knight2)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert not torch.any(game_over_tensor)
    
    def testKnightAndBishop(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        knight = chess_cpp.Piece('n', True, 0, 1)
        bishop = chess_cpp.Piece('b', False, 0, 2)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, knight)
        boards.setPiece(0, bishop)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert not torch.any(game_over_tensor)
    
    def testGameNotOverTwofoldRepetition(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        knight = chess_cpp.Piece('n', True, 0, 1)
        bishop = chess_cpp.Piece('b', False, 0, 2)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, knight)
        boards.setPiece(0, bishop)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert not torch.any(game_over_tensor)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert not torch.any(game_over_tensor)
    
    def testGameOverThreefoldOneBatch(self):
        boards = chess_cpp.BatchedBoard(False, 2, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        knight = chess_cpp.Piece('n', True, 0, 1)
        bishop = chess_cpp.Piece('b', False, 0, 2)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, knight)
        boards.setPiece(0, bishop)
        boards.setPiece(1, king1)
        boards.setPiece(1, king2)
        boards.setPiece(1, knight)
        boards.setPiece(1, bishop)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)
        assert not torch.any(game_over_tensor)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)
        assert not torch.any(game_over_tensor)
        batched_board[1][1][0][0] = 1 # Add a knight.
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)
        game_over_1 = game_over_tensor[1]
        assert not torch.any(game_over_1)
        game_over_0 = game_over_tensor[0]
        assert game_over_0[0] == False
        assert game_over_0[1] == False
        assert game_over_0[2] == False
        assert game_over_0[3] == True # Threefold repetition
        assert game_over_0[4] == False
    
    def testResetThenGameoverThreefold(self):
        boards = chess_cpp.BatchedBoard(False, 2, 0)
        dud_move_count = boards.get_starting_move_count_list().cuda()
        king1 = chess_cpp.Piece('k', False, 3, 3)
        king2 = chess_cpp.Piece('k', True, 6, 3)
        knight = chess_cpp.Piece('n', True, 0, 1)
        bishop = chess_cpp.Piece('b', False, 0, 2)
        boards.setPiece(0, king1)
        boards.setPiece(0, king2)
        boards.setPiece(0, knight)
        boards.setPiece(0, bishop)
        boards.setPiece(1, king1)
        boards.setPiece(1, king2)
        boards.setPiece(1, knight)
        boards.setPiece(1, bishop)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)
        assert not torch.any(game_over_tensor)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)
        assert not torch.any(game_over_tensor)
        dummy_game_over = torch.Tensor([True, False])
        boards.reset_repetitions(dummy_game_over)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)
        game_over_0 = game_over_tensor[0]
        game_over_1 = game_over_tensor[1]
        assert not torch.any(game_over_0)
        assert game_over_1[0] == False
        assert game_over_1[1] == False
        assert game_over_1[2] == False
        assert game_over_1[3] == True # Threefold repetition
        assert game_over_1[4] == False
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert not torch.any(game_over_tensor)
        repetition_status = get_repetition_status(boards, batched_board)
        game_over_tensor = is_game_over(batched_board, move_layer, repetition_status, dud_move_count)[0]
        assert game_over_tensor[0] == False
        assert game_over_tensor[1] == False
        assert game_over_tensor[2] == False
        assert game_over_tensor[3] == True # Threefold repetition
        assert game_over_tensor[4] == False
