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
