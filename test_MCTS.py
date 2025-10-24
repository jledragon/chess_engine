# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 19:23:16 2025

@author: jledragon
"""

# Run this with "python -m unittest test_MCTS.py"
# Run individual test with python -m unittest test_MCTS.TestMCTS.<test>
import torch
import unittest
import chess_cpp
from chess_training_loop import A2CMoveAgent, MCTSGraph
from chess_py_utils import get_human_readable_board


BATCH_SIZE = 1


class TestMCTS(unittest.TestCase):
    '''
    Unit tests for MCTS.

    Warning - These might be SLOW. Running one at a time is recommended
    and only without WSL.
    '''

    def testWinEndgameSimple(self):
        torch._dynamo.config.cache_size_limit = 64
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        queen = chess_cpp.Piece('q', True, 0, 6)
        king = chess_cpp.Piece('k', True, 6, 5)
        enemyKing = chess_cpp.Piece('k', False, 7, 7)
        boards.setPiece(0, queen)
        boards.setPiece(0, king)
        boards.setPiece(0, enemyKing)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)
        
        our_ai_agent = A2CMoveAgent(boards, batched_board, {})
        white_mcts = MCTSGraph(our_ai_agent, boards)
        # From this position, white has 27 valid moves. Set this value high enough
        # to reach terminal states - White can play mate in one.
        white_mcts.depth = 800
        white_mcts.init_top_node_if_empty_graph(batched_board, move_layer)
        #print(get_human_readable_board(white_mcts.top_node.current_board[0,:6,:,:]))
        #print(white_mcts.top_node.opponent_move_to_get_here)
        white_mcts.generate_graph()
        #for child in white_mcts.top_node.children:
            #print("")
            #print(get_human_readable_board(child.current_board[0,:6,:,:]))
            #print(child.opponent_move_to_get_here, child.opponent_move_to_get_here.shape)
        
        # Test MCTS N, W, Q and P values. Test that the 27 first-depth moves are explored before expanding any. Test what happens when we reach a winning terminal.
        #print(white_mcts.top_node.N, white_mcts.top_node.W, white_mcts.top_node.Q, len(white_mcts.top_node.children))
        #for child in white_mcts.top_node.children:
        #    print(child.N, child.W, child.Q, child.state_value, child.predicted_value, child.terminal_status)
        #print("\n")
        #print(white_mcts.top_node.P, white_mcts.top_node.P.dtype)
        #print(white_mcts.top_node.get_probability_distribution(), white_mcts.top_node.get_probability_distribution().dtype)
        #print(cross_entropy(white_mcts.top_node.P, white_mcts.top_node.get_probability_distribution()))
        #print(cross_entropy(white_mcts.top_node.P, white_mcts.top_node.P))
        #print(torch.sum(white_mcts.top_node.P), torch.sum(white_mcts.top_node.get_probability_distribution()))

    def testLoseEndgameSimple(self):
        torch._dynamo.config.cache_size_limit = 64
        boards = chess_cpp.BatchedBoard(False, BATCH_SIZE, 0)
        enemyQueen = chess_cpp.Piece('q', False, 6, 0)
        enemyKing = chess_cpp.Piece('k', False, 6, 5)
        king = chess_cpp.Piece('k', True, 7, 7)
        boards.setPiece(0, enemyQueen)
        boards.setPiece(0, king)
        boards.setPiece(0, enemyKing)
        batched_board = boards.to_tensor().cuda()
        move_layer = chess_cpp.get_moves_for_player(batched_board)

        our_ai_agent = A2CMoveAgent(boards, batched_board, {})
        white_mcts = MCTSGraph(our_ai_agent, boards)
        # From this position, white has 27 valid moves. Set this value high enough
        # to reach terminal states - White can play mate in one.
        white_mcts.depth = 800
        white_mcts.init_top_node_if_empty_graph(batched_board, move_layer)
        white_mcts.generate_graph()

        # Test MCTS N, W, Q and P values. Test that the 27 first-depth moves are explored before expanding any. Test what happens when we reach a winning terminal.
        print(white_mcts.top_node.N, white_mcts.top_node.W, white_mcts.top_node.Q, white_mcts.top_node.P, len(white_mcts.top_node.children))
        for child in white_mcts.top_node.children:
            print(child.N, child.W, child.Q, child.terminal_status)
