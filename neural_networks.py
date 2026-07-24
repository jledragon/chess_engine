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
from mobile_cvt.ConvUser import ConvUser
from chess_py_utils import get_random_move, conditional_compile, expand_all_moves
from blitz.modules import BayesianLinear, BayesianConv2d
import chess_cpp
import os
from einops.layers.torch import Reduce
from my_optim import SharedAdamW


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


class Simple2DNetwork(nn.Module):
    """
    CNN with resnets. Performs reasonably well.
    """
    
    def __init__(self, value_part):
        # The 4 conv resnet blocks setup was the best performer of this bunch for the full game.
        # See https://github.com/undera/chess-engine-nn
        super(Simple2DNetwork, self).__init__()

        # 3x3 view
        self.conv_3_1 = nn.Conv2d(6, 8, (3, 3))
        self.bn_c31 = nn.BatchNorm2d(8)
        nn.init.xavier_uniform_(self.conv_3_1.weight)
        nn.init.ones_(self.bn_c31.weight)
        nn.init.zeros_(self.conv_3_1.bias)
        nn.init.zeros_(self.bn_c31.bias)
        self.conv_3_2 = nn.Conv2d(8, 16, (3, 3))
        self.bn_c32 = nn.BatchNorm2d(16)
        nn.init.xavier_uniform_(self.conv_3_2.weight)
        nn.init.ones_(self.bn_c32.weight)
        nn.init.zeros_(self.conv_3_2.bias)
        nn.init.zeros_(self.bn_c32.bias)
        self.conv_3_3 = nn.Conv2d(16, 32, (3, 3))
        self.bn_c33 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.conv_3_2.weight)
        nn.init.ones_(self.bn_c32.weight)
        nn.init.zeros_(self.conv_3_2.bias)
        nn.init.zeros_(self.bn_c32.bias)

        # 4x4 view
        self.conv_4_1 = nn.Conv2d(6, 8, (4, 4))
        self.bn_c41 = nn.BatchNorm2d(8)
        nn.init.xavier_uniform_(self.conv_4_1.weight)
        nn.init.ones_(self.bn_c41.weight)
        nn.init.zeros_(self.conv_4_1.bias)
        nn.init.zeros_(self.bn_c41.bias)
        self.conv_4_2 = nn.Conv2d(8, 16, (4, 4))
        self.bn_c42 = nn.BatchNorm2d(16)
        nn.init.xavier_uniform_(self.conv_4_2.weight)
        nn.init.ones_(self.bn_c42.weight)
        nn.init.zeros_(self.conv_4_2.bias)
        nn.init.zeros_(self.bn_c42.bias)

        # 5x5 view
        self.conv_5_1 = nn.Conv2d(6, 8, (5, 5))
        self.bn_c51 = nn.BatchNorm2d(8)
        nn.init.xavier_uniform_(self.conv_5_1.weight)
        nn.init.ones_(self.bn_c51.weight)
        nn.init.zeros_(self.conv_5_1.bias)
        nn.init.zeros_(self.bn_c51.bias)
        self.conv_5_2 = nn.Conv2d(8, 16, (3, 3))
        self.bn_c52 = nn.BatchNorm2d(16)
        nn.init.xavier_uniform_(self.conv_5_2.weight)
        nn.init.ones_(self.bn_c52.weight)
        nn.init.zeros_(self.conv_5_2.bias)
        nn.init.zeros_(self.bn_c52.bias)

        # 6x6 view
        self.conv_6_1 = nn.Conv2d(6, 8, (6, 6))
        self.bn_c61 = nn.BatchNorm2d(8)
        nn.init.xavier_uniform_(self.conv_6_1.weight)
        nn.init.ones_(self.bn_c61.weight)
        nn.init.zeros_(self.conv_6_1.bias)
        nn.init.zeros_(self.bn_c61.bias)

        # 7x7 view
        self.conv_7_1 = nn.Conv2d(6, 8, (7, 7))
        self.bn_c71 = nn.BatchNorm2d(8)
        nn.init.xavier_uniform_(self.conv_7_1.weight)
        nn.init.ones_(self.bn_c71.weight)
        nn.init.zeros_(self.conv_7_1.bias)
        nn.init.zeros_(self.bn_c71.bias)

        # 8x8 view
        self.conv_8_1 = nn.Conv2d(6, 8, (8, 8))
        self.bn_c81 = nn.BatchNorm2d(8)
        nn.init.xavier_uniform_(self.conv_8_1.weight)
        nn.init.ones_(self.bn_c81.weight)
        nn.init.zeros_(self.conv_8_1.bias)
        nn.init.zeros_(self.bn_c81.bias)

        self.y_layer = nn.Linear(368, 4_096)
        nn.init.xavier_uniform_(self.y_layer.weight)
        nn.init.zeros_(self.y_layer.bias)

        self.value_part = value_part
        # TODO - rethink promotions. They are currently very inefficient with parameters.
        #self.prom_layer = nn.Linear(8192, 4 * 4096)
        if value_part:
            self.v1_layer = nn.Linear(368, 20)
            self.bn_v1 = nn.BatchNorm1d(20)
            self.v2_layer = nn.Linear(20, 1)
            nn.init.xavier_uniform_(self.v1_layer.weight)
            nn.init.xavier_uniform_(self.v2_layer.weight)
            nn.init.ones_(self.bn_v1.weight)
            nn.init.zeros_(self.v1_layer.bias)
            nn.init.zeros_(self.v2_layer.bias)
            nn.init.zeros_(self.bn_v1.bias)
        self.lr = 1e-4
    
    def set_train_mode(self):
        self.train()

    def set_test_mode(self):
        self.eval()

    def get_parameters(self):
        return self.parameters()

    def forward(self, board):
        promo = torch.tensor([[0, 0, 0, 1]]).repeat(board.shape[0], 1).to(torch.float32).cuda() # Queen

        # 3x3 view
        xh_3x3_1 = F.relu(self.bn_c31(self.conv_3_1(board.to(torch.float32))))
        xh_3x3_2 = F.relu(self.bn_c32(self.conv_3_2(xh_3x3_1)))
        xh_3x3_3 = F.relu(self.bn_c33(self.conv_3_3(xh_3x3_2)))
        xh_3x3 = xh_3x3_3.reshape(xh_3x3_3.shape[0], 128)

        # 4x4 view
        xh_4x4_1 = F.relu(self.bn_c41(self.conv_4_1(board.to(torch.float32))))
        xh_4x4_2 = F.relu(self.bn_c42(self.conv_4_2(xh_4x4_1)))
        xh_4x4 = xh_4x4_2.reshape(xh_4x4_2.shape[0], 64)

        # 5x5 view
        xh_5x5_1 = F.relu(self.bn_c51(self.conv_5_1(board.to(torch.float32))))
        xh_5x5_2 = F.relu(self.bn_c52(self.conv_5_2(xh_5x5_1)))
        xh_5x5 = xh_5x5_2.reshape(xh_5x5_2.shape[0], 64)

        # 6x6 view
        xh_6x6 = F.relu(self.bn_c61(self.conv_6_1(board.to(torch.float32))))
        xh_6x6 = xh_6x6.reshape(xh_6x6.shape[0], 72)

        # 7x7 view
        xh_7x7 = F.relu(self.bn_c71(self.conv_7_1(board.to(torch.float32))))
        xh_7x7 = xh_7x7.reshape(xh_7x7.shape[0], 32)

        # 8x8 view
        xh_8x8 = F.relu(self.bn_c81(self.conv_8_1(board.to(torch.float32))))
        xh_8x8 = xh_8x8.reshape(xh_8x8.shape[0], 8)

        # Features will all levels of view
        xh = torch.cat((xh_3x3, xh_4x4, xh_5x5, xh_6x6, xh_7x7, xh_8x8), axis=1)
        out = self.y_layer(xh)
        #promo = self.prom_layer(xh3)
        #promo = promo.reshape(xh2.shape[0], 4096, 4)
        if not self.value_part:
            return out, promo
        else:
            hv = F.relu(self.bn_v1(self.v1_layer(xh)))
            v_out = self.v2_layer(hv)
            v = torch.tanh(v_out)
            return out, promo, v


class CvTNetwork(ConvUser):
    """
    Use MobileCVT without compression.
    """
    
    def __init__(self, value_part):
        super(CvTNetwork, self).__init__()
        self.conv_1 = self.conv_nxn_bn(6, 64, kernel_size=3, stride=1)
        patch_size=(2, 2)
        expansion = 4
        dropout = 0.1
        self.lr = 1e-6

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(64, 80, 1, expansion),
            MobileCvTBlock(
                96, 2, 80,
                kernel_size=3, patch_size=patch_size, dropout=dropout)
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(80, 96, 1, expansion),
            MobileCvTBlock(
                120, 4, 96,
                kernel_size=3, patch_size=patch_size, dropout=dropout)
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(96, 112, 1, expansion),
            MobileCvTBlock(
                144, 3, 112,
                kernel_size=3, patch_size=patch_size, dropout=dropout)
        ]))

        self.to_logits = nn.Sequential(
            self.conv_1x1_bn(112, 4096),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(4096, 4096, bias=False)
        )

    def set_train_mode(self):
        self.train()

    def set_test_mode(self):
        self.eval()

    def get_parameters(self):
        return self.parameters()

    def forward(self, board):
        promo = torch.tensor([[0, 0, 0, 1]]).repeat(board.shape[0], 1).to(torch.float32).cuda()
        x = self.conv_1(board.to(torch.float32))

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x), promo


class SimpleLinearNetwork(nn.Module):
    """
    For debugging.
    """

    def __init__(self, value_part):
        super(SimpleLinearNetwork, self).__init__()
        self.x_layer = nn.Linear(384, 4096)
        self.h_layer = nn.Linear(4096, 4096)
        self.y_layer = nn.Linear(4096, 4096)
        self.prom_layer = nn.Linear(4096, 4 * 4096)
        self.bn_1 = nn.BatchNorm1d(4096)
        self.bn_2 = nn.BatchNorm1d(4096)
        self.prom_layer = nn.Linear(4096, 4 * 4096)
        self.value_part = value_part
        nn.init.xavier_uniform_(self.x_layer.weight)
        nn.init.xavier_uniform_(self.h_layer.weight)
        nn.init.xavier_uniform_(self.y_layer.weight)
        if value_part:
            self.hidden_v_layer = nn.Linear(4096, 512)
            nn.init.xavier_uniform_(self.hidden_v_layer.weight)
            self.bn_3 = nn.BatchNorm1d(512)
            self.v_layer = nn.Linear(512, 1)
        self.lr = 1e-4
    
    def set_train_mode(self):
        self.train()

    def set_test_mode(self):
        self.eval()
    
    def get_parameters(self):
        return self.parameters()
    
    def forward(self, board):
        board = board.reshape(board.shape[0], 384)
        #promo = torch.tensor([[0, 0, 0, 1]]).repeat(board.shape[0], 1).to(torch.float32).cuda()
        xh = F.relu(self.bn_1(self.x_layer(board.to(torch.float32))))
        hh = F.relu(self.bn_2(self.h_layer(xh)))
        out = self.y_layer(hh)
        promo = self.prom_layer(hh)
        promo = promo.reshape(hh.shape[0], 4096, 4)
        if not self.value_part:
            return out, promo
        else:
            hv = F.relu(self.bn_3(self.hidden_v_layer(xh)))
            hv_1 = hv.reshape(hv.shape[0], 512)
            v = torch.tanh(self.v_layer(hv_1))
            return out, promo, v


class DQNChessNetwork:
    def __init__(self):
        self.eps = 0.9
        self.prev_eps = self.eps
        self.chess_network = Simple2DNetwork(False).cuda()
        self.qnet_network = Simple2DNetwork(False).cuda()
        try:
            self.chess_network = torch.compile(self.chess_network)
            self.qnet_network = torch.compile(self.qnet_network)
        except RuntimeError:
            print("Warning - compile not supported.")
        self.qnet_network.set_test_mode()
        self.softmax = nn.Softmax(dim=1)
        self.tau = 0.999
        self.discount_factor = 0.99
        self.lr = self.chess_network.lr
        self.optimiser = torch.optim.AdamW(self.chess_network.get_parameters(), lr=self.lr, weight_decay=1e-5, amsgrad=True)
        self.huberLoss_function = nn.SmoothL1Loss()
        #self.MSELoss_function = nn.MSELoss()
        #self.cross_entropy = nn.CrossEntropyLoss()
    
    @conditional_compile
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
        # torch.nn.utils.clip_grad_norm_(self.chess_network.parameters(), 0.1)  # Alternative.
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


class A2CChessNetwork:
    def __init__(self):
        self.chess_network = Simple2DNetwork(True).cuda()
        #self.chess_network = SimpleLinearNetwork(True).cuda()
        if os.environ.get('try_compile', 'False').lower() == 'true':
            self.chess_network = torch.compile(self.chess_network)
        self.lr = self.chess_network.lr  # All use the same class, so all lr should be the same.
        self.optimiser = SharedAdamW(self.chess_network.get_parameters(), lr=self.lr, weight_decay=1e-5, amsgrad=True)
        self.optimiser.share_memory()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.train_mode = True
        self.gamma = 0.99
        self.gae_lambda = 1.0
        self.entropy_coef = 0.01
        self.value_loss_coef = 2.0
        self.num_iter_delta = 200
        self.num_iters_to_train = 400

    @conditional_compile
    def get_move_logits(self, out_move, ml_mask):
        filtered_out = torch.where(
            ml_mask,
            out_move,
            -float('inf'),
        )
        return filtered_out
    
    @conditional_compile
    def get_model_move_and_state(self, board):
        out_move, out_prom, state_value = self.chess_network.forward(board)
        return out_move, out_prom, state_value
    
    @conditional_compile
    def get_mcts_moves(self, current_board, out_move, out_prom, move_layer):
        assert current_board.shape[1] == 8, "Remember to use the full board here."
        ml_mask = move_layer.to(torch.bool).reshape((move_layer.shape[0], move_layer.shape[1] * move_layer.shape[2]))
        num_moves_per_batch_element = torch.sum(move_layer, dim=(1,2), dtype=torch.int32)
        move_logits = self.get_move_logits(out_move, ml_mask)
        log_move_probs = self.log_softmax(move_logits)
        finite_log_probs = torch.isfinite(log_move_probs)
        valid_move_indices = torch.argwhere(finite_log_probs)[:,1:]
        valid_probs = torch.exp(log_move_probs[ml_mask])
        # TODO - rethink promotions. They are currently too inefficient with NN parameters.
        # 8x4 encoding "to" column context only would be preferable to 4096x4 encoding full from/to square context
        #valid_promotion_probs_per_move = out_prom[ml_mask]  # old
        #val_prom = self.softmax(valid_promotion_probs_per_move)  # old
        val_prom = out_prom.repeat(valid_probs.shape[0], 1)  # new
        expanded_boards, expanded_moves, expanded_promotions, expanded_valid_probs, new_splits = expand_all_moves(current_board, val_prom, valid_move_indices, valid_probs, num_moves_per_batch_element)
        all_expanded_moves = []
        all_expanded_promotions = []
        all_expanded_valid_probs = []
        total = 0
        # Return on a per-node basis.
        for split in new_splits:
            all_expanded_moves.append(expanded_moves[total:total+split])
            all_expanded_promotions.append(expanded_promotions[total:total+split])
            all_expanded_valid_probs.append(expanded_valid_probs[total:total+split])
            total += split
        return all_expanded_moves, all_expanded_promotions, all_expanded_valid_probs
    
    @conditional_compile
    def get_best_opponent_move(self, board, move_layer):
        # Opponent move for MCTS. Assume promotions to Queens
        out_move, out_prom, _ = self.chess_network.forward(board)
        out_prom = out_prom.to(torch.int8)
        ml_mask = move_layer.to(torch.bool).reshape((move_layer.shape[0], move_layer.shape[1] * move_layer.shape[2]))
        move_logits = self.get_move_logits(out_move, ml_mask)
        move_probs = self.log_softmax(move_logits)
        action = torch.argmax(move_probs, dim=1, keepdim=True)
        ft1 = action // 64
        ft2 = action % 64
        f1 = ft1 // 8
        t1 = ft1 % 8
        f2 = ft2 // 8
        t2 = ft2 % 8
        nn_move = torch.cat((f1, t1, f2, t2), dim=1).to(torch.int8)
        return nn_move, out_prom
    
    def update_network(self, state, graph_probs, final_game_value):
        pred_act, pred_prom, pred_value = self.get_model_move_and_state(state[:,:6,:,:])
        move_layer = chess_cpp.get_moves_for_player(state)
        cross_entropy_total = 0
        _, _, action_probs = self.get_mcts_moves(state, pred_act, pred_prom, move_layer)
        valid_batches = 0
        for i, ap in enumerate(action_probs):
            ap_logits = torch.logit(ap)
            if not len(ap) == 1 and torch.all(torch.isfinite(ap_logits)):
                # In cases there there was not only one move.
                valid_batches += 1
                cross_entropy_total = cross_entropy_total + F.cross_entropy(torch.logit(ap), graph_probs[i])
        if valid_batches > 0:
            cross_entropy_mean = cross_entropy_total / valid_batches
        else:
            cross_entropy_mean = 0  # safety
        reward_loss = F.mse_loss(torch.squeeze(pred_value, 1), final_game_value)
        combined_loss = cross_entropy_mean + 5 * reward_loss
        print(f"Cross entropy loss: {cross_entropy_mean}, Reward loss: {reward_loss}")
        self.optimiser.zero_grad()
        combined_loss.backward()
        self.optimiser.step()

    def set_train_mode(self):
        self.chess_network.set_train_mode()
        self.train_mode = True
    
    def set_test_mode(self):
        self.chess_network.set_test_mode()
        self.train_mode = False

    def update_training_params(self, num_logged_games):
        if self.num_iters_to_train < 2_000:
            self.num_iters_to_train += self.num_iter_delta

    def training_session(self, start_epoch, num_logged_games, memory):
        """
        Learn on the data that we have gathered so far.
        """
        self.set_train_mode()
        current_epoch = self.train_on_data(start_epoch, memory)
        self.update_training_params(num_logged_games)
        self.set_test_mode()
        return current_epoch

    def train_on_data(self, start_epoch, memory):
        print(f"Training for {self.num_iters_to_train} iterations...")
        for current_epoch in range(start_epoch, start_epoch + self.num_iters_to_train):
            states, mcts_probs, game_vals = memory.sample_training_batch()
            try:
                self.update_network(states, mcts_probs, game_vals)
            except Exception as e:
                # Training seems to be error prone - save models and data if an error occurred
                print("Something went wrong during training...")
                memory.save_data()
                self.save_models()
                raise e
        return current_epoch

    def save_models(self):
        torch.save({
            "model_state": self.chess_network.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
        }, 'models/last_model')
    
    def load_models(self, purpose):
        if not os.path.exists('models/last_model'):
            print("Warning - saved model files not found.")
        else:
            last_model = torch.load('models/last_model')
            self.chess_network.load_state_dict(last_model["model_state"])
            self.optimiser.load_state_dict(last_model["optimiser_state"])
            if purpose == 'train':
                self.chess_network.train()
            elif purpose == 'eval':
                self.chess_network.eval()
