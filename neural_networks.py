# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:34:44 2023

@author: jledragon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mobile_cvt.MV2Block import MV2Block
from mobile_cvt.MobileCvTBlock import MobileCvTBlock
from mobile_cvt.MobileCvT import MobileCvT
from chess_py_utils import get_random_move, get_human_readable_board, compile_if_supported
from blitz.modules import BayesianLinear, BayesianConv2d
from itertools import chain
import chess_cpp
import os


class ResnetBlockFC(nn.Module):
    '''
    Fully connected ResNet Block class.
    '''
    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResnetBlockFC, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)
        self.bn_1 = nn.BatchNorm1d(size_in)
        self.bn_2 = nn.BatchNorm1d(size_in)
        nn.init.xavier_uniform_(self.fc_0.weight)
        #nn.init.xavier_uniform_(self.fc_1.weight)

    def forward(self, x):
        net = self.actvn(self.fc_0(x))
        dx = self.fc_1(net)

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return self.actvn(x_s + dx)


class ResnetBlockFC2D(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResnetBlockFC2D, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Conv2d(size_in, size_h, (3, 3), padding=1)
        self.fc_1 = nn.Conv2d(size_h, size_out, (3, 3), padding=1)
        self.bn_1 = nn.BatchNorm2d(size_in)
        self.bn_2 = nn.BatchNorm2d(size_in)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(size_in, size_out, 3, padding=1, bias=False)
        # Initialization
        nn.init.xavier_uniform_(self.fc_0.weight)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.actvn(self.bn_1(self.fc_0(x)))
        dx = self.bn_2(self.fc_1(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return self.actvn(x_s + dx)


class FullChessNetwork(nn.Module):
    def __init__(self):
        super(FullChessNetwork, self).__init__()
        self.mobile_vit = MobileCvT(
            image_size=(8, 8),
            dims=[96, 120, 144],
            #channels=[16, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024],
            channels=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
            num_classes=128,  # Encoded board dimension
            dataset_dim=6
        )
        self.fc1 = ResnetBlockFC(128)
        self.fc2 = ResnetBlockFC(128)
        self.fc3 = ResnetBlockFC(size_in=128, size_out=4096)
        self.fc4 = ResnetBlockFC(size_in = 4096, size_h=1024, size_out=256)
        self.fc5 = ResnetBlockFC(size_in = 256, size_h=64, size_out=4)
    
    def set_train_mode(self):
        self.train()

    def set_test_mode(self):
        self.eval()
    
    def get_parameters(self):
        return self.parameters()
    
    def forward(self, board):
        promo = torch.tensor([[0, 0, 0, 1]]).repeat(256, 1).to(torch.float32).cuda()
        enc = self.mobile_vit(board.to(torch.float32))
        split = self.fc1(enc)
        out = self.fc2(split)
        out = self.fc3(out)
        #promo = self.fc4(split)
        #promo = self.fc5(promo)
        return out, promo


class Simple2DNetwork(nn.Module):
    
    def __init__(self):
        # The 4 conv resnet blocks setup was the best performer of this bunch for the full game.
        super(Simple2DNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(6, 256, (3, 3), padding=1) # 8
        #self.conv_2 = nn.Conv2d(256, 512, 3, padding=0) # 6
        #self.conv_3 = nn.Conv2d(512, 1024, 3, padding=0) # 4
        #self.conv_4 = nn.Conv2d(1024, 2048, 3, padding=0) # 2
        #self.conv_5 = nn.Conv2d(256, 1024, 5, padding=0)
        #self.conv_6 = nn.Conv2d(512, 2048, 5, padding=0)
        #self.conv_7 = nn.Conv2d(256, 2048, 7, padding=0)
        self.conv_8 = nn.Conv2d(256, 128, (1, 1), padding=0)
        #self.conv_5 = nn.Conv2d(2048, 4096, 2, padding=0)
        self.bn_1 = nn.BatchNorm2d(256)
        self.bn_2 = nn.BatchNorm2d(128)
        self.conv_block_1 = ResnetBlockFC2D(256, 256)  # TODO - split the Bayesian version into its own class.
        self.conv_block_2 = ResnetBlockFC2D(256, 256)
        self.conv_block_3 = ResnetBlockFC2D(256, 256)
        self.conv_block_4 = ResnetBlockFC2D(256, 256)
        #self.m_block_1 = MV2Block(256, 256, 1, 4)
        #self.m_block_2 = MV2Block(256, 256, 1, 4)
        #self.m_block_3 = MV2Block(256, 256, 1, 4)
        #self.t_block_1 = MobileCvTBlock(96, 2, 256, 3, (2, 2))
        #self.t_block_2 = MobileCvTBlock(120, 4, 256, 3, (2, 2))
        #self.t_block_3 = MobileCvTBlock(144, 3, 256, 3, (2, 2))
        #self.res_block_1 = ResnetBlockFC(4096, 4096)
        #self.res_block_2 = ResnetBlockFC(4096, 4096)
        self.y_layer = nn.Linear(8192, 4096)
        nn.init.xavier_uniform_(self.conv_1.weight)
        #nn.init.xavier_uniform_(self.conv_2.weight)
        #nn.init.xavier_uniform_(self.conv_3.weight)
        #nn.init.xavier_uniform_(self.conv_4.weight)
        #nn.init.xavier_uniform_(self.conv_5.weight)
        #nn.init.xavier_uniform_(self.conv_6.weight)
        #nn.init.xavier_uniform_(self.conv_7.weight)
        nn.init.xavier_uniform_(self.conv_8.weight)
        nn.init.xavier_uniform_(self.y_layer.weight)
        #self.test = nn.Linear(4096, 4096)
        #self.test.weight = torch.nn.Parameter(torch.ones((4096, 4096)))
        #self.test.bias = torch.nn.Parameter(torch.zeros((4096)))
        #self.test.eval()
    
    def set_train_mode(self):
        self.train()
        #self.test.eval()

    def set_test_mode(self):
        self.eval()
    
    def get_parameters(self):
        return self.parameters()
    
    def forward(self, board):
        promo = torch.tensor([[0, 0, 0, 1]]).repeat(256, 1).to(torch.float32).cuda()
        xh1 = F.relu(self.bn_1(self.conv_1(board.to(torch.float32))))
        #xh2 = self.t_block_1(self.m_block_1(xh1))
        #xh3 = self.t_block_2(self.m_block_2(xh2))
        #xh4 = self.t_block_3(self.m_block_3(xh3))
        
        #xh2 = F.relu(self.conv_2(xh1))
        #xh3 = F.relu(self.conv_3(xh2)) # + self.conv_5(xh1))
        #xh4 = F.relu(self.conv_4(xh3)) # + self.conv_6(xh2)) # + self.conv_7(xh1))
        
        xh2 = self.conv_block_1(xh1)
        xh3 = self.conv_block_2(xh2)
        xh4 = self.conv_block_3(xh3)
        xh5 = self.conv_block_4(xh4)
        xh6 = F.relu(self.bn_2(self.conv_8(xh5)))
        xh7 = xh6.reshape(xh6.shape[0], 8192)
        out = self.y_layer(xh7)
        #out = self.test(out)
        return out, promo
    


class SimpleLinearNetwork(nn.Module):
    """
    For debugging.
    """

    def __init__(self):
        super(SimpleLinearNetwork, self).__init__()
        self.x_layer = nn.Linear(384, 4096)
        self.h_layer = nn.Linear(4096, 4096)
        #self.h1_layer = ResnetBlockFC(4096, 4096)
        #self.h2_layer = ResnetBlockFC(4096, 4096)
        #self.h3_layer = ResnetBlockFC(4096, 4096)
        #self.h4_layer = ResnetBlockFC(4096, 4096)
        self.y_layer = nn.Linear(4096, 4096)
        nn.init.xavier_uniform_(self.x_layer.weight)
        nn.init.xavier_uniform_(self.h_layer.weight)
        nn.init.xavier_uniform_(self.y_layer.weight)
        # The capstone layer exists so that all trainable weights have an impact on all moves.
        self.capstone = nn.Linear(4096, 4096)
        self.capstone.weight = torch.nn.Parameter(torch.ones((4096, 4096)))
        self.capstone.bias = torch.nn.Parameter(torch.zeros((4096)))
        self.capstone.eval()
    
    def set_train_mode(self):
        self.train()
        self.capstone.eval()

    def set_test_mode(self):
        self.eval()
    
    def get_parameters(self):
        return self.parameters()
    
    def forward(self, board):
        board = board.reshape(board.shape[0], 384)
        promo = torch.tensor([[0, 0, 0, 1]]).repeat(256, 1).to(torch.float32).cuda()
        xh = F.relu(self.x_layer(board.to(torch.float32)))
        hh = self.h_layer(xh)
        #hh = self.h1_layer(hh)
        out = self.y_layer(hh)
        out = self.capstone(out)
        return out, promo


class DQNChessNetwork:
    def __init__(self):
        self.eps = 0.9
        self.prev_eps = self.eps
        self.chess_network = Simple2DNetwork().cuda()
        self.qnet_network = Simple2DNetwork().cuda()
        try:
            self.chess_network = torch.compile(self.chess_network)
            self.qnet_network = torch.compile(self.qnet_network)
        except RuntimeError:
            print("Warning - compile not supported.")
        #self.chess_network = SimpleLinearNetwork().cuda()
        #self.qnet_network = SimpleLinearNetwork().cuda()
        #self.chess_network = FullChessNetwork().cuda()
        #self.qnet_network = FullChessNetwork().cuda()
        self.qnet_network.set_test_mode()
        self.softmax = nn.Softmax(dim=1)
        self.tau = 0.999
        self.discount_factor = 0.99
        self.lr = 1e-4
        self.optimiser = torch.optim.AdamW(self.chess_network.get_parameters(), lr=self.lr, weight_decay=1e-5, amsgrad=True)
        self.huberLoss_function = nn.SmoothL1Loss()
        #self.MSELoss_function = nn.MSELoss()
        #self.cross_entropy = nn.CrossEntropyLoss()
    
    @compile_if_supported
    def get_allowed_max(self, out_move, move_layer):
        filtered_out = torch.where(
            move_layer.reshape((move_layer.shape[0], move_layer.shape[1] * move_layer.shape[2])) == 1,
            out_move,
            -float('inf'),
        )
        sm_filtered_out = self.softmax(filtered_out)
        max_filtered_out = torch.argmax(sm_filtered_out, dim=1, keepdim=True)
        return max_filtered_out
    
    def get_move(self, board, move_layer):
        out_move, out_prom = self.chess_network.forward(board)
        max_filtered_out = self.get_allowed_max(out_move, move_layer)
        sm_prom = self.softmax(out_prom)
        max_out_prom = torch.argmax(sm_prom, dim=1)
        nn_prom = F.one_hot(max_out_prom, num_classes=4).to(torch.int8)
        ft1 = max_filtered_out // 64
        ft2 = max_filtered_out % 64
        f1 = ft1 // 8
        t1 = ft1 % 8
        f2 = ft2 // 8
        t2 = ft2 % 8
        nn_move = torch.cat((f1, t1, f2, t2), dim=1).to(torch.int8)
        random_moves, random_promotions = get_random_move(board, move_layer)
        eps_tensor = torch.rand((board.shape[0])).cuda().unsqueeze(1)
        eps_accounted_moves = torch.where(
            eps_tensor > self.eps,
            nn_move,
            random_moves
        )
        eps_accounted_promotions = torch.where(
            eps_tensor > self.eps,
            nn_prom,
            random_promotions
        )
        return eps_accounted_moves, eps_accounted_promotions

    def update_network(self, states, actions, rewards, terminals, next_states):
        moves, promotions = actions
        moves = moves.to(torch.long)
        rewards = rewards.to(torch.float32)
        flat_moves = (moves[:,0] * 8 + moves[:,1]) * 64 + moves[:,2] * 8 + moves[:,3]
        qs, _ = self.chess_network.forward(states)
        qsa = torch.gather(qs, 1, flat_moves.unsqueeze(1))
        qs_next, _ = self.qnet_network.forward(next_states)
        # It is better all around to compute the move_layer yet again, rather than storing this huge value in the experience buffer.
        move_layer = chess_cpp.get_moves_for_player(next_states)
        max_next = self.get_allowed_max(qs_next, move_layer)
        qsa_next = torch.gather(qs_next, 1, max_next)
        not_terminals = 1 - terminals
        qsa_next_target = rewards.unsqueeze(1) + not_terminals.unsqueeze(1) * self.discount_factor * qsa_next
        loss_targ = qsa_next_target.detach()
        q_network_loss = self.huberLoss_function(qsa, loss_targ)
        self.optimiser.zero_grad()
        q_network_loss.backward()
        torch.nn.utils.clip_grad_value_(self.chess_network.parameters(), 100)
        self.optimiser.step()
        
    
    def soft_target_update(self):
        network_params = self.chess_network.get_parameters()
        target_params = self.qnet_network.get_parameters()
        for net_params, targ_params in zip(network_params, target_params):
            targ_params.data.copy_(targ_params.data * self.tau + net_params.data * (1 - self.tau))
    
    def set_train_mode(self):
        self.chess_network.set_train_mode()
    
    def set_test_mode(self):
        self.chess_network.set_test_mode()
    
    def save_models(self):
        torch.save({
            "model_state": self.chess_network.state_dict(),
            "target_state": self.qnet_network.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
        }, 'models/last_model')
    
    def load_models(self, purpose):
        if not os.path.exists('models/last_model'):
            print("Warning - saved model files not found.")
        else:
            last_model = torch.load('models/last_model')
            self.chess_network.load_state_dict(last_model["model_state"])
            self.qnet_network.load_state_dict(last_model["target_state"])
            self.optimiser.load_state_dict(last_model["optimiser_state"])
            self.qnet_network.eval()
            if purpose == 'train':
                self.chess_network.train()
            elif purpose == 'eval':
                self.chess_network.eval()
