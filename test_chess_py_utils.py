# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:03:02 2023

@author: jledragon
"""

# Run this with "python -m unittest test_chess_py_utils.py"
import torch
import unittest
import chess_cpp
from chess_py_utils import get_game_value_for_white


BATCH_SIZE = 1


class TestUtils(unittest.TestCase):
    '''
    Unit tests for chess_py_utils
    '''
    
    def testGameValueForWhite(self):
        game_over_tensor = torch.Tensor([[False, False, True, False, False]]).to(torch.bool).cuda()
        colour_tensor = torch.Tensor([True]).to(torch.bool).cuda()
        major_outcomes = game_over_tensor[0, 0:3]
        true_game_value = get_game_value_for_white(major_outcomes, colour_tensor[0])
        assert true_game_value == 0
        
        game_over_tensor = torch.Tensor([[False, True, False, False, False]]).to(torch.bool).cuda()
        colour_tensor = torch.Tensor([True]).to(torch.bool).cuda()
        major_outcomes = game_over_tensor[0, 0:3]
        true_game_value = get_game_value_for_white(major_outcomes, colour_tensor[0])
        assert true_game_value == 0
        
        game_over_tensor = torch.Tensor([[True, False, False, False, False]]).to(torch.bool).cuda()
        colour_tensor = torch.Tensor([True]).to(torch.bool).cuda()
        major_outcomes = game_over_tensor[0, 0:3]
        true_game_value = get_game_value_for_white(major_outcomes, colour_tensor[0])
        assert true_game_value == 1
        
        game_over_tensor = torch.Tensor([[True, False, False, False, False]]).to(torch.bool).cuda()
        colour_tensor = torch.Tensor([False]).to(torch.bool).cuda()
        major_outcomes = game_over_tensor[0, 0:3]
        true_game_value = get_game_value_for_white(major_outcomes, colour_tensor[0])
        assert true_game_value == -1

    def test_selecting_promotions(self):
        """ Test indexing promotions from a 4x8 NN output. """
        nn_out = torch.Tensor([
            [
                [0.1, 0.1, 0.1, 0.7],
                [0.1, 0.1, 0.2, 0.6],
                [0.1, 0.1, 0.3, 0.5],
                [0.1, 0.2, 0.1, 0.6], ###
                [0.1, 0.3, 0.1, 0.5],
                [0.2, 0.1, 0.1, 0.6],
                [0.3, 0.1, 0.1, 0.5],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.2, 0.2, 0.2, 0.4],
                [0.2, 0.2, 0.21, 0.39],
                [0.2, 0.2, 0.22, 0.38],
                [0.2, 0.21, 0.2, 0.39],
                [0.2, 0.22, 0.2, 0.38], ###
                [0.21, 0.2, 0.2, 0.39],
                [0.22, 0.2, 0.2, 0.38],
                [0.25, 0.25, 0.25, 0.25],
            ],
            [
                [0.3, 0.3, 0.3, 0.1],
                [0.3, 0.3, 0.31, 0.09], ###
                [0.3, 0.3, 0.32, 0.08],
                [0.3, 0.31, 0.3, 0.09],
                [0.3, 0.32, 0.3, 0.08],
                [0.31, 0.3, 0.3, 0.09],
                [0.32, 0.3, 0.3, 0.08],
                [0.25, 0.25, 0.25, 0.25],
            ]
        ]).cuda()
        select = torch.Tensor([[3], [4], [1]]).to(torch.int64).cuda() # Hypothetical t2
        nn_select_2d = torch.index_select(nn_out, 1, select[:,0])
        nn_select = torch.transpose(torch.diagonal(nn_select_2d, dim1=0, dim2=1), 0, 1)
        assert torch.equal(nn_select, torch.Tensor([[0.1, 0.2, 0.1, 0.6], [0.2, 0.22, 0.2, 0.38], [0.3, 0.3, 0.31, 0.09]]).cuda())
