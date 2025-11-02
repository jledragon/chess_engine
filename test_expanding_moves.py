# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:15:05 2024

@author: jledragon
"""

# Run this with "python -m unittest test_expanding_moves.py"
import torch
import unittest
import chess_cpp
from chess_py_utils import expand_all_moves


BATCH_SIZE = 1


class TestExpandingMoves(unittest.TestCase):
    '''
    Unit tests for expanding all possible moves.
    '''

    def testPromoteMovesDetection(self):
        # python -m unittest test_expanding_moves.TestExpandingMoves.testPromoteMovesDetection
        boards = chess_cpp.BatchedBoard(False, 10, 0)
        pawn_1 = chess_cpp.Piece('p', True, 6, 0, False, False)
        pawn_2 = chess_cpp.Piece('p', True, 6, 1, False, False)
        pawn_3 = chess_cpp.Piece('p', True, 6, 2, False, False)
        pawn_4 = chess_cpp.Piece('p', True, 6, 3, False, False)
        pawn_5 = chess_cpp.Piece('p', True, 6, 4, False, False)
        pawn_6 = chess_cpp.Piece('p', True, 6, 5, False, False)
        pawn_7 = chess_cpp.Piece('p', True, 6, 6, False, False)
        pawn_8 = chess_cpp.Piece('p', True, 6, 7, False, False)
        pawn_9 = chess_cpp.Piece('p', True, 5, 5, False, False)
        king_0 = chess_cpp.Piece('k', True, 0, 0)
        king_1 = chess_cpp.Piece('k', False, 0, 7)
        boards.setPiece(0, pawn_9)
        boards.setPiece(1, pawn_1)
        boards.setPiece(2, pawn_2)
        boards.setPiece(3, pawn_3)
        boards.setPiece(4, pawn_4)
        boards.setPiece(5, pawn_5)
        boards.setPiece(6, pawn_6)
        boards.setPiece(7, pawn_7)
        boards.setPiece(8, pawn_8)
        for i in range(0, 10):
            boards.setPiece(i, king_0)
            boards.setPiece(i, king_1)

        batched_board = boards.to_tensor().cuda()
        dummy_moves = torch.Tensor([
            [5, 5, 6, 5],
            [6, 0, 7, 0],
            [6, 1, 7, 1],
            [6, 2, 7, 2],
            [6, 3, 7, 3],
            [6, 4, 7, 4],
            [6, 5, 7, 5],
            [6, 6, 7, 6],
            [6, 7, 7, 7],
            [0, 0, 0, 1],
        ]).to(torch.int8).cuda()
        is_promotion = chess_cpp.get_pawn_promote_move_mask(batched_board, dummy_moves)
        assert torch.equal(is_promotion, torch.Tensor([False, True, True, True, True, True, True, True, True, False]).cuda())
    
    def testExpandPromotions(self):
        promotion_mask = torch.Tensor([False, True, True, True, True, True, True, True, True, False]).to(torch.bool).cuda()
        promotions = torch.Tensor([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]).to(torch.int8).cuda()
        new_promotions = chess_cpp.expand_promotions(promotions, promotion_mask)
        assert torch.equal(new_promotions, torch.Tensor([
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]).to(torch.int8).cuda())

    def testExpandBoards(self):
        boards = chess_cpp.BatchedBoard(False, 10, 0)
        pawn_1 = chess_cpp.Piece('p', True, 6, 0, False, False)
        pawn_2 = chess_cpp.Piece('p', True, 6, 1, False, False)
        pawn_3 = chess_cpp.Piece('p', True, 6, 2, False, False)
        pawn_4 = chess_cpp.Piece('p', True, 6, 3, False, False)
        pawn_5 = chess_cpp.Piece('p', True, 6, 4, False, False)
        pawn_6 = chess_cpp.Piece('p', True, 6, 5, False, False)
        pawn_7 = chess_cpp.Piece('p', True, 6, 6, False, False)
        pawn_8 = chess_cpp.Piece('p', True, 6, 7, False, False)
        pawn_9 = chess_cpp.Piece('p', True, 5, 5, False, False)
        king_0 = chess_cpp.Piece('k', True, 0, 0)
        king_1 = chess_cpp.Piece('k', False, 0, 7)
        boards.setPiece(0, pawn_9)
        boards.setPiece(1, pawn_1)
        boards.setPiece(2, pawn_2)
        boards.setPiece(3, pawn_3)
        boards.setPiece(4, pawn_4)
        boards.setPiece(5, pawn_5)
        boards.setPiece(6, pawn_6)
        boards.setPiece(7, pawn_7)
        boards.setPiece(8, pawn_8)
        for i in range(0, 10):
            boards.setPiece(i, king_0)
            boards.setPiece(i, king_1)

        batched_board = boards.to_tensor().cuda()
        promotion_mask = torch.Tensor([False, True, True, True, True, True, True, True, True, False]).to(torch.bool).cuda()
        new_boards = chess_cpp.expand_boards(batched_board, promotion_mask)
        assert new_boards.shape[0] == 34
        assert torch.equal(batched_board[0], new_boards[0])
        for i in range(1, 9):
            for j in range(0, 4):
                assert torch.equal(batched_board[i], new_boards[4*i-j])
        assert torch.equal(batched_board[9], new_boards[33])

    def testExpandAllMoves(self):
        boards = chess_cpp.BatchedBoard(False, 1, 0)
        pawn_1 = chess_cpp.Piece('p', True, 6, 0, False, False)
        king_0 = chess_cpp.Piece('k', True, 1, 0)
        king_1 = chess_cpp.Piece('k', False, 0, 7)
        boards.setPiece(0, pawn_1)
        boards.setPiece(0, king_0)
        boards.setPiece(0, king_1)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        move_layer_for_board = move_layer[0]
        flat_moves = move_layer_for_board.reshape((move_layer_for_board.shape[0] * move_layer_for_board.shape[1]))
        legal_moves = torch.where(flat_moves == 1)[0].unsqueeze(1)
        valid_moves = torch.ones((6)).cuda() / 6
        prom = torch.Tensor([1, 0, 0, 0]).to(torch.float32).cuda()
        expanded_boards, expanded_moves, expanded_promotions, expanded_valid_probs = expand_all_moves(batched_board[0], prom, legal_moves, valid_moves)
        assert torch.equal(expanded_promotions, torch.Tensor([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]).to(torch.bool).cuda())
        assert torch.all(torch.isclose(expanded_valid_probs, torch.Tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0, 0, 0]).cuda()))
        assert expanded_boards.shape == (9, 8, 8, 8)
        for i in range(0, 9):
            assert torch.equal(expanded_boards[i], batched_board[0])
