# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:53:53 2023

@author: jledragon
"""

# Run this with "python -m unittest test_blocks.py"
import torch
import unittest
import chess_cpp

BATCH_SIZE = 3


class TestBlocks(unittest.TestCase):
    '''
    Unit tests for blocks and pins.
    '''
    
    def test_no_king(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        relevant_board = block_squares[1][3*8 + 4].reshape((8, 8))
        assert torch.sum(relevant_board) == 64 # All allowed.
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_down(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 7, 3)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[4][3] == 1
        assert relevant_board[5][3] == 1
        assert relevant_board[6][3] == 1
        assert torch.sum(relevant_board) == 4
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_up(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 0, 3)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[2][3] == 1
        assert relevant_board[1][3] == 1
        assert torch.sum(relevant_board) == 3
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_left(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 3, 0)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[3][2] == 1
        assert relevant_board[3][1] == 1
        assert torch.sum(relevant_board) == 3
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_right(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 3, 7)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[3][4] == 1
        assert relevant_board[3][5] == 1
        assert relevant_board[3][6] == 1
        assert torch.sum(relevant_board) == 4
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_friend_in_way_vert(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 7, 3)
        knight = chess_cpp.Piece('n', True, 5, 3)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # I am pinned. I cannot move.
        relevant_board = block_squares[1][5*8 + 3].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[4][3] == 1
        assert relevant_board[5][3] == 1
        assert relevant_board[6][3] == 1
        assert torch.sum(relevant_board) == 4
        # Can be any square, as long as it's not the attacker or King
        relevant_board_2 = block_squares[1][5*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board_2, torch.ones((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_enemy_in_way_vert(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 7, 3)
        dark_knight = chess_cpp.Piece('n', False, 5, 3)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        boards.setPiece(1, dark_knight)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # I am pinned. I cannot move (even though I'm an enemy - this is relative to "me" and "my" moves).
        relevant_board = block_squares[1][5*8 + 3].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[4][3] == 1
        assert relevant_board[5][3] == 1
        assert relevant_board[6][3] == 1
        assert torch.sum(relevant_board) == 4
        # Can be any square, as long as it's not the attacker or King
        relevant_board_2 = block_squares[1][5*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board_2, torch.ones((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_friend_in_way_horiz(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 3, 7)
        knight = chess_cpp.Piece('n', True, 3, 5)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # I am pinned. I cannot move.
        relevant_board = block_squares[1][3*8 + 5].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[3][4] == 1
        assert relevant_board[3][5] == 1
        assert relevant_board[3][6] == 1
        assert torch.sum(relevant_board) == 4
        # Can be any square, as long as it's not the attacker or King
        relevant_board_2 = block_squares[1][5*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board_2, torch.ones((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_rook_enemy_in_way_horiz(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        rook = chess_cpp.Piece('r', False, 3, 3)
        king = chess_cpp.Piece('k', True, 3, 7)
        dark_knight = chess_cpp.Piece('n', False, 3, 5)
        boards.setPiece(1, rook)
        boards.setPiece(1, king)
        boards.setPiece(1, dark_knight)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # I am pinned. I cannot move (even though I'm an enemy - this is relative to "me" and "my" moves).
        relevant_board = block_squares[1][3*8 + 5].reshape((8, 8))
        assert relevant_board[3][3] == 1
        assert relevant_board[3][4] == 1
        assert relevant_board[3][5] == 1
        assert relevant_board[3][6] == 1
        assert torch.sum(relevant_board) == 4
        # Can be any square, as long as it's not the attacker or King
        relevant_board_2 = block_squares[1][5*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board_2, torch.ones((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_double_attack_rook(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        rook1 = chess_cpp.Piece('r', False, 3, 7)
        rook2 = chess_cpp.Piece('r', False, 7, 3)
        boards.setPiece(1, king)
        boards.setPiece(1, rook1)
        boards.setPiece(1, rook2)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Double attack - no moves.
        relevant_board = block_squares[1][3*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board, torch.zeros((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_bishop_northeast(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        bishop = chess_cpp.Piece('b', False, 6, 0)
        boards.setPiece(1, bishop)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[4][2] == 1
        assert relevant_board[5][1] == 1
        assert relevant_board[6][0] == 1
        assert torch.sum(relevant_board) == 3
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_bishop_northwest(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        bishop = chess_cpp.Piece('b', False, 7, 7)
        boards.setPiece(1, bishop)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[4][4] == 1
        assert relevant_board[5][5] == 1
        assert relevant_board[6][6] == 1
        assert relevant_board[7][7] == 1
        assert torch.sum(relevant_board) == 4
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_bishop_southeast(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        bishop = chess_cpp.Piece('b', False, 0, 0)
        boards.setPiece(1, bishop)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[2][2] == 1
        assert relevant_board[1][1] == 1
        assert relevant_board[0][0] == 1
        assert torch.sum(relevant_board) == 3
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_bishop_southwest(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        bishop = chess_cpp.Piece('b', False, 0, 6)
        boards.setPiece(1, bishop)
        boards.setPiece(1, king)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Can be any square, as long as it's not the attacker or King
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[2][4] == 1
        assert relevant_board[1][5] == 1
        assert relevant_board[0][6] == 1
        assert torch.sum(relevant_board) == 3
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_bishop_friend_in_way(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', False, 7, 7)
        king = chess_cpp.Piece('k', True, 3, 3)
        knight = chess_cpp.Piece('n', True, 5, 5)
        boards.setPiece(1, bishop)
        boards.setPiece(1, king)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # I am pinned. I cannot move.
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[4][4] == 1
        assert relevant_board[5][5] == 1
        assert relevant_board[6][6] == 1
        assert relevant_board[7][7] == 1
        assert torch.sum(relevant_board) == 4
        # Can be any square, as long as it's not the attacker or King
        relevant_board_2 = block_squares[1][6*8 + 0].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board_2, torch.ones((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_valid_block_squares_bishop_enemy_in_way(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        bishop = chess_cpp.Piece('b', False, 7, 7)
        king = chess_cpp.Piece('k', True, 3, 3)
        dark_knight = chess_cpp.Piece('n', False, 5, 5)
        boards.setPiece(1, bishop)
        boards.setPiece(1, king)
        boards.setPiece(1, dark_knight)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # I am pinned. I cannot move.
        relevant_board = block_squares[1][5*8 + 5].reshape((8, 8))
        assert relevant_board[4][4] == 1
        assert relevant_board[5][5] == 1
        assert relevant_board[6][6] == 1
        assert relevant_board[7][7] == 1
        assert torch.sum(relevant_board) == 4
        # Can be any square, as long as it's not the attacker or King
        relevant_board_2 = block_squares[1][6*8 + 0].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board_2, torch.ones((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_double_attack_bishop(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        bishop1 = chess_cpp.Piece('b', False, 7, 7)
        bishop2 = chess_cpp.Piece('b', False, 0, 6)
        boards.setPiece(1, king)
        boards.setPiece(1, bishop1)
        boards.setPiece(1, bishop2)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Double attack - no moves.
        relevant_board = block_squares[1][3*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board, torch.zeros((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_double_attack_rook_bishop(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        rook = chess_cpp.Piece('r', False, 3, 7)
        bishop = chess_cpp.Piece('b', False, 0, 6)
        boards.setPiece(1, king)
        boards.setPiece(1, rook)
        boards.setPiece(1, bishop)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Double attack - no moves.
        relevant_board = block_squares[1][3*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board, torch.zeros((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_knight_attack(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        knight = chess_cpp.Piece('n', False, 5, 4)
        boards.setPiece(1, king)
        boards.setPiece(1, knight)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        relevant_board = block_squares[1][6*8 + 2].reshape((8, 8)).cpu()
        assert relevant_board[5][4] == 1
        assert torch.sum(relevant_board) == 1
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_pawn_attack(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        pawn = chess_cpp.Piece('p', False, 4, 4)
        boards.setPiece(1, king)
        boards.setPiece(1, pawn)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        relevant_board = block_squares[1][6*8 + 2].reshape((8, 8)).cpu()
        assert relevant_board[4][4] == 1
        assert torch.sum(relevant_board) == 1
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_king_attack(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        pawn = chess_cpp.Piece('k', False, 3, 4)
        boards.setPiece(1, king)
        boards.setPiece(1, pawn)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        relevant_board = block_squares[1][6*8 + 2].reshape((8, 8)).cpu()
        assert relevant_board[3][4] == 1
        assert torch.sum(relevant_board) == 1
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_hybrid_attack_1(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        knight = chess_cpp.Piece('n', False, 5, 4)
        rook = chess_cpp.Piece('r', False, 3, 7)
        boards.setPiece(1, king)
        boards.setPiece(1, knight)
        boards.setPiece(1, rook)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Double attack - no moves.
        relevant_board = block_squares[1][3*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board, torch.zeros((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
    
    def test_hybrid_attack_2(self):
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        king = chess_cpp.Piece('k', True, 3, 3)
        pawn = chess_cpp.Piece('p', False, 4, 4)
        bishop = chess_cpp.Piece('b', False, 0, 6)
        boards.setPiece(1, king)
        boards.setPiece(1, pawn)
        boards.setPiece(1, bishop)
        batched_board = boards.to_tensor().cuda()
        block_squares = chess_cpp.get_valid_blocking_squares_player(batched_board)
        # Double attack - no moves, even if already blocking one threat.
        relevant_board = block_squares[1][1*8 + 5].reshape((8, 8)).cpu()
        assert torch.equal(relevant_board, torch.zeros((8, 8), dtype=torch.int8))
        
        irrelevant_1 = block_squares[0].cpu()
        irrelevant_2 = block_squares[2].cpu()
        assert torch.equal(irrelevant_1, torch.ones((64, 64), dtype=torch.int8))
        assert torch.equal(irrelevant_2, torch.ones((64, 64), dtype=torch.int8))
