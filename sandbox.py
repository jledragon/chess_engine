# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:54:32 2024

@author: jledragon
"""

import torch

# A place to play around with random experiments.


def get_all_binary_moves(batch_size):
    all_moves = torch.arange(0, 4096).unsqueeze(0)
    all_moves = all_moves.repeat(batch_size, 1)
    mask = 2**torch.arange(11, -1, -1).to(all_moves.device, all_moves.dtype)
    return all_moves.unsqueeze(-1).bitwise_and(mask).ne(0).float()


if __name__ == '__main__':
    all_mov = get_all_binary_moves(256)
    print(all_mov[0,0,:], all_mov[0,1,:], all_mov[0,2,:])
