# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:39:05 2023

@author: jledragon

Run this file with environment "jle".
"""

import time
import torch
import chess_cpp
import argparse
from stockfish import Stockfish
from random import random
from abc import ABC, abstractmethod
import platform
import numpy as np
from datetime import datetime as dt
from pathlib import Path
from chess_py_utils import (
    get_repetition_status,
    is_game_over,
    get_random_move,
    reset_move_counts,
    flip_board,
    possibly_reset_game,
    reset_colour_list,
    get_white_view,
    convert_jle_to_UCI_notation,
    convert_UCI_to_jle_notation,
    get_human_readable_board,
    flip_episode,
    get_firepower_score,
    conditional_compile,
    get_game_over_message,
    get_game_value_for_white,
    save_full_game_artifacts,
    #get_moves_for_player_with_reuse
)
from neural_networks import DQNChessNetwork, A2CChessNetwork

BATCH_SIZE = 1  # 1 for A2C, 256 for DQN

class AIMoveAgent(ABC):
    """
    Any and all autonomous agents should inherit this interface.
    """

    @abstractmethod
    def decide_move(self, board_state):
        """
        Choose a move given the board state.
        """
        pass
    
    @abstractmethod
    def enact_move(self, move, board_state):
        """
        Play the chosen move forward
        """
        pass

    def decide_and_enact_move(self, board_state):
        """
        Play a full turn.
        """
        move = self.decide_move(board_state)
        move, state = self.enact_move(move, board_state)
        return move, state


class JLEAIMoveAgent(AIMoveAgent, ABC):
    """
    An AI that can play against Stockfish.
    """
    
    @abstractmethod
    def prepare_for_training(self):
        pass
    
    @abstractmethod
    def prepare_for_evaluation(self):
        pass
    
    @abstractmethod
    def save_all_models(self):
        pass
    
    @abstractmethod
    def load_all_models(self):
        pass

    @abstractmethod
    def train_step(self, epoch):
        pass

    @abstractmethod
    def self_play_and_training_session(self, boards, start_epoch):
        pass

    def log_stockfish_move(self, move, board_state, starting_colour_me):
        """
        Record to the JLE game state what the Stockfish move was.
        """
        jle_move, jle_promotion = convert_UCI_to_jle_notation(move, starting_colour_me)
        return self.enact_move((jle_move, jle_promotion), board_state)
    
    @conditional_compile
    def enact_move(self, move, board_state, full_game_over_conditions=True):
        # If full_game_over_conditions is False, skip ongoing draw conditions such as 50-move rule and threefold repetition.
        # Some algorithms need to explore, so these should be disabled during exploration.
        selected_move, promotion = move
        board_tensor, colour_list, dud_move_count = board_state
        if not full_game_over_conditions:
            dud_move_count = torch.zeros((board_tensor.shape[0], 1)).to(board_tensor.device).to(torch.int8)
        # Do the move
        chess_cpp.enact_moves(board_tensor, selected_move, promotion, dud_move_count)
        # Flip the board to opponent view
        flipped_board, inv_colour_list = flip_board(board_tensor, colour_list)
        # Log this position (white's view) for threefold repetition check
        white_view = get_white_view(board_tensor, flipped_board, colour_list)
        if full_game_over_conditions:
            repetition_status = get_repetition_status(self.boards, white_view)
        else:
            repetition_status = torch.zeros((board_tensor.shape[0], 1)).to(board_tensor.device).to(torch.bool)
        colour_list = inv_colour_list
        # Get opponent's move to see if the game is over by the move we just did
        opponent_move_layer = chess_cpp.get_moves_for_player(flipped_board)
        # Check whether the game is over under any condition
        game_over_tensor = is_game_over(flipped_board, opponent_move_layer, repetition_status, dud_move_count)
        game_over = torch.any(game_over_tensor, dim=1)
        # Reset threefold repetitions list, move counts and colour wherever the game is over
        if full_game_over_conditions:
            self.boards.reset_repetitions(game_over)
        reset_move_counts(dud_move_count, game_over)
        reset_colour_list(colour_list, game_over)
        # Flip the boards, reset wherever the game is over
        board_tensor = flipped_board
        board_tensor = possibly_reset_game(board_tensor, game_over, self.starting_position)
        
        return (selected_move, promotion), (dud_move_count, board_tensor, colour_list, opponent_move_layer, game_over_tensor)

    @conditional_compile
    def reset_all(self, board_state):
        board_tensor, colour_list, dud_move_count = board_state
        game_over_all = torch.ones((board_tensor.shape[0])).cuda().to(torch.bool)
        self.boards.reset_repetitions(game_over_all)
        reset_move_counts(dud_move_count, game_over_all)
        reset_colour_list(colour_list, game_over_all)
        board_tensor = possibly_reset_game(board_tensor, game_over_all, self.starting_position)
        return board_tensor, colour_list, dud_move_count


class StockfishMoveAgent(AIMoveAgent):
    """
    A wrapper for a Stockfish player.
    """

    def __init__(self, boards, starting_position):
        self.boards = boards
        self.starting_position = starting_position
        if platform.system() == 'Linux':
            stockfish_path = "./stockfish/stockfish-ubuntu-x86-64-avx2"
        elif platform.system() == 'Windows':
            stockfish_path = r"C:\Users\jledragon\Documents\Chess_21\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2"
        self.stockfish = Stockfish(stockfish_path)
    
    def start_new_game(self):
        """
        Reset Stockfish's game state for the current game.'
        """
        self.moves_so_far = []
        self.stockfish.set_position([])
    
    def set_elo(self, elo):
        """
        Set the level of difficulty for Stockfish.
        """
        self.stockfish.set_elo_rating(elo)
    
    def decide_move(self, board_state):
        board_tensor = board_state
        assert board_tensor.shape[0] == 1
        assert len(board_tensor.shape) == 4
        best_move = self.stockfish.get_best_move()
        return best_move
    
    def enact_move(self, move, board_state):
        self.moves_so_far.append(move)
        self.stockfish.set_position(self.moves_so_far)
        return move, board_state
    
    def log_opponent_move(self, move, promotion, board_tensor, starting_colour_opponent):
        """
        Log the move of the opponent AI to Stockfish.
        """
        uci_move = convert_jle_to_UCI_notation(move, promotion, board_tensor, not starting_colour_opponent)
        self.moves_so_far.append(uci_move)
        try:
            self.stockfish.set_position(self.moves_so_far)
        except Exception as e:
            print("Stockfish's version of the game is:\n")
            print(self.stockfish.get_board_visual())
            print("JLE version of the game is:\n")
            print(get_human_readable_board(board_tensor[0], not starting_colour_opponent))
            print("Moves taken to get here:\n")
            print(self.moves_so_far)
            raise e


class DQNExperienceBuffer:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size  # multiplied by batch size.
        self.state_buffer = torch.empty((0, 6, 8, 8)).to(torch.int8)
        self.move_buffer = torch.empty((0, 4)).to(torch.int8)
        self.promotion_buffer = torch.empty((0, 4)).to(torch.int8)
        self.rewards_buffer = torch.empty((0)).to(torch.int8)
        self.terminals_buffer = torch.empty((0)).to(torch.int8)
        self.next_state_buffer = torch.empty((0, 6, 8, 8)).to(torch.int8)
        self.training_batch_size = batch_size

    def add_to_buffer(self, state, action, rewards, terminals, next_state):
        move, promotion = action
        if self.state_buffer.shape[0] > ((self.max_size + 1) * state.shape[0]):
            begin = state.shape[0]
        else:
            begin = 0
        self.state_buffer = torch.cat((self.state_buffer[begin:], state), dim=0)
        self.move_buffer = torch.cat((self.move_buffer[begin:], move), dim=0)
        self.promotion_buffer = torch.cat((self.promotion_buffer[begin:], promotion), dim=0)
        self.rewards_buffer = torch.cat((self.rewards_buffer[begin:], rewards), dim=0)
        self.terminals_buffer = torch.cat((self.terminals_buffer[begin:], terminals), dim=0)
        self.next_state_buffer = torch.cat((self.next_state_buffer[begin:], next_state), dim=0)
    
    def sample_training_batch(self):
        rand_sample_indices = torch.randint(low=0, high=self.state_buffer.shape[0], size=(self.training_batch_size,))
        states = self.state_buffer[rand_sample_indices].cuda()
        moves = self.move_buffer[rand_sample_indices].cuda()
        promotions = self.promotion_buffer[rand_sample_indices].cuda()
        rewards = self.rewards_buffer[rand_sample_indices].cuda()
        terminals = self.terminals_buffer[rand_sample_indices].cuda()
        next_states = self.next_state_buffer[rand_sample_indices].cuda()
        return states, (moves, promotions), rewards, terminals, next_states


class DQNMoveAgent(JLEAIMoveAgent):
    """
    A neural network, reinforcement learning-based algorithm to choose moves.
    """

    def __init__(self, boards, starting_position, enabled_optional_rewards):
        self.boards = boards
        self.starting_position = starting_position
        self.previous_state = None
        self.previous_action = None
        self.previous_rewards = None
        self.previous_terminals = None
        self.current_opponent_state = None
        self.opponent_action = None
        self.opponent_rewards = None
        self.opponent_terminals = None
        self.experience_buffer = DQNExperienceBuffer(10_000, BATCH_SIZE)
        self.q_network = DQNChessNetwork()
        self.win_reward = 300 # 100
        self.lose_reward = -100 # -100
        self.move_reward = -1.2 # -1
        self.update_rate = 40 # 40
        self.move_reward_factor = 6 # 6
        self.enabled_optional_rewards = enabled_optional_rewards
    
    @conditional_compile
    def _firepower_score_fn(self, board_state):
        if "firepower" in self.enabled_optional_rewards:
            my_firepower = get_firepower_score(board_state, False)
            opponent_firepower = get_firepower_score(board_state, True)
            scaled_difference_firepower = (my_firepower - opponent_firepower) * 0.1
            score = 2 * torch.sigmoid(scaled_difference_firepower) - 1  # Range between -1 and 1.
            return score
        else:
            return torch.zeros(board_state.shape[0]).to(board_state.device)
    
    @conditional_compile
    def _move_reward_fn(self, opponent_move_layer):
        if "num_moves" in self.enabled_optional_rewards:
            sum_moves = torch.sum(opponent_move_layer, (1, 2))
            move_reward = self.move_reward_factor / (torch.sqrt(sum_moves + 1))
            # Do not count checkmate rewards twice in parallel with win/lose rewards.
            move_reward = torch.where(sum_moves == 0, 0, move_reward)
            return move_reward
        else:
            return torch.zeros(opponent_move_layer.shape[0]).to(opponent_move_layer.device)

    @conditional_compile
    def _firepower_per_num_moves_fn(self, game_state_bundle):
        board_state, opponent_move_layer = game_state_bundle
        if "firepower_per_num_moves" in self.enabled_optional_rewards:
            opponent_firepower = get_firepower_score(board_state, True) + 4  # 4 is the King's firepower.
            sum_moves = torch.sum(opponent_move_layer, (1, 2))
            fp_np = torch.sqrt(opponent_firepower / (sum_moves + 1))
            return fp_np
        else:
            return torch.zeros(board_state.shape[0]).to(board_state.device)
    
    @conditional_compile
    def _get_my_rewards(self, game_over_tensor, game_state_bundle):
        board_state, opponent_move_layer = game_state_bundle
        # In this case we get partial rewards. Punishments for losing are doled out at the next step (update_opponent_rewards).
        rewards = torch.where(game_over_tensor[:,0], self.win_reward, self.move_reward)
        # Getting your opponent into trouble gets a reward.
        num_moves_reward = self._move_reward_fn(opponent_move_layer)
        firepower_reward = self._firepower_score_fn(board_state)
        firepower_per_num_moves_reward = self._firepower_per_num_moves_fn(game_state_bundle) / 100.0
        rewards = rewards + num_moves_reward + firepower_reward + firepower_per_num_moves_reward
        return rewards
    
    def apply_all_rewards(self, game_over_tensor, game_state_bundle):
        rewards = self._get_my_rewards(game_over_tensor, game_state_bundle)
        self._update_opponent_rewards(game_over_tensor, game_state_bundle)
        return rewards

    @conditional_compile
    def _update_opponent_rewards(self, game_over_tensor, game_state_bundle):
        board_state, opponent_move_layer = game_state_bundle
        if self.opponent_rewards is not None:
            losses_opp = game_over_tensor[:, 0].cpu()
            num_moves_reward = -self._move_reward_fn(opponent_move_layer)
            punishment = num_moves_reward.cpu()
            self.opponent_rewards += punishment
            if torch.sum(losses_opp) > 0:
                self.opponent_terminals[losses_opp] = True
                self.opponent_rewards[losses_opp] = self.lose_reward
    
    def prepare_for_training(self):
        self.q_network.set_train_mode()
        self.q_network.eps = self.q_network.prev_eps
    
    def prepare_for_evaluation(self):
        self.q_network.set_test_mode()
        self.q_network.prev_eps = self.q_network.eps
        self.q_network.eps = 0

    def store_current_artifacts(self, current_state, action, rewards, terminals):
        self.previous_state = self.current_opponent_state
        self.previous_action = self.opponent_action
        self.previous_rewards = self.opponent_rewards
        self.previous_terminals = self.opponent_terminals
        self.current_opponent_state = current_state
        self.opponent_action = action
        self.opponent_rewards = rewards
        self.opponent_terminals = terminals
    
    def store_training_artifacts(self, dqn_store_bundle, opponent_move_layer):
        current_state, move, promotion, rewards, game_over = dqn_store_bundle
        if self.previous_state is not None:
            self.experience_buffer.add_to_buffer(self.previous_state, self.previous_action, self.previous_rewards, self.previous_terminals, current_state[:,:6,:,:].cpu())
        # Store state, actions, rewards and terminals for the next pass
        self.store_current_artifacts(current_state[:,:6,:,:].cpu(), (move.cpu(), promotion.cpu()), (rewards.cpu()), game_over.cpu())
    
    def train_step(self, epoch, augment_data=True):
        for update in range(0, self.update_rate):
            states, actions, rewards, terminals, next_states = self.experience_buffer.sample_training_batch()
            if augment_data:
                # Doubles our data by flipping horizontally, which is always invariant for this chess setup.
                states, actions, next_states = flip_episode(states, actions, next_states)
            # TODO - randomly augment data by flipping horizontally
            self.q_network.update_network(states, actions, rewards, terminals, next_states)
            self.q_network.soft_target_update()
        # Training strategy here
        if self.q_network.eps > 0.05:
            self.q_network.eps = self.q_network.eps * 0.998 # 0.9942
        else:
            self.q_network.eps = 0.05
        if epoch == 2100:
            for group in self.q_network.optimiser.param_groups:
                group['lr'] = group['lr'] * 0.1

    def self_play_and_training_session(self, boards, start_epoch):
        """
        Play an AI against itself and train on past data.
        """
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((BATCH_SIZE)).to(torch.bool).cuda()
        total_games = 0
        
        #now = time.time()
        end_epoch = start_epoch + 100
        for move_num in range(start_epoch, end_epoch):
            self.q_network.set_test_mode()
            current_state = batched_board.clone()
            (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
                self.decide_and_enact_move((batched_board, colour_list, dud_move_count))
            rewards = self.apply_all_rewards(game_over_tensor, (batched_board, opponent_move_layer))
            game_over = torch.any(game_over_tensor, dim=1)
            # Store state, action, rewards, terminals and next state to the buffer.
            self.store_training_artifacts((current_state, move, promotion, rewards, game_over), opponent_move_layer)
            if move_num >= 100:
                self.q_network.set_train_mode()
                self.train_step(move_num)
            games_done = torch.sum(game_over.to(torch.int8))
            total_games += games_done
            print(move_num)
        #elapsed = time.time() - now
        #print(elapsed)
        print(self.q_network.eps, start_epoch)
        return end_epoch

    def decide_move(self, board_state):
        board_tensor, _, _ = board_state
        # Get the moves
        move_layer = chess_cpp.get_moves_for_player(board_tensor)
        # Choose a move
        dqn_move, dqn_promotion = self.q_network.get_move(board_tensor[:,:6,:,:], move_layer)
        return (dqn_move, dqn_promotion)
    
    def save_all_models(self):
        # Later - with these two methods, make them more configurable.
        self.q_network.save_models()
    
    def load_all_models(self):
        self.q_network.load_models('train')


class MCTSNode:
    """
    A single node in MCTS representing a state and its attributes for MCTS.
    """
    
    def __init__(self, current_state, moves_for_state, terminal_status):
        self.N = 0  # Number of times an action has been taken from this state.
        self.W = 0  # Value of the next state.
        self.Q = 0  # Mean value of the next state.
        self.P = None  # Prior probabilities of selecting each action.
        self.valid_actions = None
        self.valid_promotions = None
        self.children = []
        # Required values for rollout below
        self.is_leaf = True
        self.rollout_done = False
        self.current_board = current_state
        self.moves_for_state = moves_for_state
        self.terminal_status = terminal_status
        self.model_move = None
        self.pred_value = None
        self.predicted_value = None
        self.model_promotion = None
        self.opponent_move_to_get_here = None  # Will be from the opponent's point of view. Does not handle promotions currently.
        self.temperature = 2
        self.opponent_actual_move = None
        self.marked_for_generation = False  # An attempt to make graph generation faster.
        self.action_probs_with_grad = None
        self.state_value = None  # Information about predicted value or better - may be GT terminal
        self.marked_for_rollout = False
    
    def update_actions(self, valid_actions, valid_promotions, action_probs):
        self.P = action_probs
        self.valid_actions = valid_actions
        self.valid_promotions = valid_promotions

    def update_values(self, state_value, predicted_value):
        self.state_value = state_value
        self.predicted_value = predicted_value
    
    def add_predictions(self, model_move, model_promotion, pred_value, opponent_move_to_get_here):
        # Add the predictions that led us to this state. This saves us a lot of computation time by batching up model predictions.
        self.model_move = model_move
        self.model_promotion = model_promotion
        self.pred_value = pred_value
        self.opponent_move_to_get_here = opponent_move_to_get_here
    
    def set_non_leaf(self):
        self.is_leaf = False
        #self.current_board = None
        self.moves_for_state = None
        self.model_move = None
        self.pred_value = None
    
    def set_opponent_actual_move(self, actual_move):
        self.opponent_actual_move = actual_move
    
    def get_probability_distribution(self):
        child_ns = [child.N for child in self.children]
        temperature_values = [(n / self.N) ** self.temperature for n in child_ns]
        sum_temp = sum(temperature_values)
        norm_temp = torch.Tensor([t / sum_temp for t in temperature_values]).to(self.P.device).to(self.P.dtype)
        return norm_temp


class MCTSGraph:
    """
    A graph of the Monte Carlo Tree Search (MCTS) for the current game, with the
    current state as the top-level node. This graph should be wiped when starting
    a new game with A2C.
    """
    
    def __init__(self, agent, boards):
        self.top_node = None
        self.depth = 800  # Hard coded.
        self.agent = agent
        self.model = agent.model
        self.boards = boards
        self.value_mask = torch.tensor([1, -1, 0]).to(torch.float32).cuda()
        self.batch_size = 256

    def reset_graph(self):
        self.top_node = None
    
    #@conditional_compile
    def init_top_node_if_empty_graph(self, current_board, moves_for_state):
        if self.top_node is None:
            not_terminal = torch.zeros((3)).to(torch.bool).cuda()  # No win, no loss, no draw
            self.top_node = MCTSNode(current_board, moves_for_state, not_terminal)

            # Special - only do this for the first move of the game.
            self.top_node.rollout_done = True
            if torch.sum(moves_for_state) > 0:
                # Possible no grad here.
                model_moves, model_proms, state_value = self.model.get_model_move_and_state(current_board[:,:6,:,:])
                self.top_node.add_predictions(None, None, state_value, None)  # No opponent move led us here.
                valid_actions, valid_promotions, action_probs = self.model.get_mcts_moves(current_board, model_moves, model_proms, moves_for_state)
                predicted_value = state_value.squeeze(0)
                self.top_node.update_actions(valid_actions[0], valid_promotions[0], action_probs[0])
                self.top_node.update_values(predicted_value, predicted_value)
    
    #@conditional_compile
    def rollout(self, nodes):
        # "Rollout" compared to traditional MCTS means predicts the value from the model, unless the ground truth terminal gives better information.
        if len(nodes) == 0:
            return
        all_boards = []
        all_moves = []
        all_model_proms = []
        all_moves_for_state = []
        for node in nodes:
            all_boards.append(node.current_board)
            all_moves.append(node.model_move)
            all_model_proms.append(node.model_promotion)
            all_moves_for_state.append(node.moves_for_state)
        all_boards = torch.cat(all_boards, dim=0)
        all_moves = torch.cat(all_moves, dim=0)
        all_model_proms = torch.cat(all_model_proms, dim=0)
        all_moves_for_state = torch.cat(all_moves_for_state, dim=0)
        valid_actions, valid_promotions, action_probs = self.model.get_mcts_moves(all_boards, all_moves, all_model_proms, all_moves_for_state)
        for ni, node in enumerate(nodes):
            node.update_actions(valid_actions[ni], valid_promotions[ni], action_probs[ni])
            node.rollout_done = True

    #@conditional_compile
    def add_state_value(self, node):
        # Deal with the current state's value.
        state_value = self.value_mask[node.terminal_status]
        state_value = node.pred_value if state_value.shape[0] == 0 else state_value
        assert not state_value.shape[0] > 1, f"Error - node end state had more than one outcome. Terminal status for node: {node.terminal_status}"
        node.update_values(state_value, node.pred_value)

    #@conditional_compile
    def batched_child_move(self, index_boards, index_actions, index_promotions, index_colour_list):
        (_, _), (_, board_tensor_half, index_colour_list, opponent_move_layer, game_over_tensor_1) = \
            self.agent.enact_move((index_actions, index_promotions), (index_boards, index_colour_list, None), full_game_over_conditions=False)
        we_won = torch.unsqueeze(game_over_tensor_1[:, 0], 1)
        we_drew_part = torch.any(game_over_tensor_1[:, 1:3], axis=1)  # Only account for stalemates and too few pieces as draws
    
        # Find and enact the best opponent move to complete the state transition.
        best_opponent_move, best_opponent_promotion = self.model.get_best_opponent_move(board_tensor_half[:,:6,:,:], opponent_move_layer)
        (_, _), (_, next_board_tensor, index_colour_list, my_next_move_layer, game_over_tensor_2) = \
            self.agent.enact_move((best_opponent_move, best_opponent_promotion), (board_tensor_half, index_colour_list, None), full_game_over_conditions=False)
        # If we have won or drawn, play out the opponent move anyway (which will be nonsense, but efficient compute-wise), but do not register a loss or draw.
        game_over_tensor_2[we_won.repeat(1, game_over_tensor_2.shape[1])] = False
        game_over_tensor_2[torch.unsqueeze(we_drew_part, 1).repeat(1, game_over_tensor_2.shape[1])] = False
        we_lost = torch.logical_and(torch.unsqueeze(game_over_tensor_2[:, 0], 1), torch.logical_not(we_won))
        we_drew = torch.unsqueeze(torch.logical_or(we_drew_part, torch.any(game_over_tensor_2[:, 1:3], axis=1)), 1)

        terminals = torch.cat((we_won, we_lost, we_drew), axis=1)
        pred_moves, pred_proms, pred_values = self.model.get_model_move_and_state(next_board_tensor[:,:6,:,:])
        return next_board_tensor, my_next_move_layer, terminals, pred_moves, pred_proms, pred_values, best_opponent_move

    #@conditional_compile  # Somehow slows things down?
    def generate_children(self, nodes):
        if len(nodes) == 0:
            return
        total_num_actions = 0
        all_current_boards = []
        all_valid_actions = []
        all_promotions = []
        all_colour_list = []
        for node in nodes:
            current_board = node.current_board
            valid_actions = node.valid_actions
            all_valid_actions.append(valid_actions)
            all_promotions.append(node.valid_promotions)
            num_actions = valid_actions.shape[0]
            total_num_actions += num_actions

            # Enact each action on copies of the board.
            multi_board = current_board.repeat(num_actions, 1, 1, 1)
            colour_list = torch.ones((num_actions)).to(torch.bool).cuda()  # We are always us, or "white" when considering expansion.
            all_current_boards.append(multi_board)
            all_colour_list.append(colour_list)
        all_current_boards = torch.cat(all_current_boards, dim=0)
        all_valid_actions = torch.cat(all_valid_actions, dim=0)
        all_promotions = torch.cat(all_promotions, dim=0)
        all_colour_list = torch.cat(all_colour_list, dim=0)
        next_boards = []
        next_move_layers = []
        all_terminals = []
        all_pred_moves = []
        all_pred_proms = []
        all_pred_values = []
        best_opponent_moves = []
        # Keep the child node computation up to some maximum batch size
        for ind_ in range(0, total_num_actions, self.batch_size):
            index_boards = all_current_boards[ind_:ind_+self.batch_size,:,:,:]
            index_actions = all_valid_actions[ind_:ind_+self.batch_size,:]
            index_promotions = all_promotions[ind_:ind_+self.batch_size,:]
            index_colour_list = all_colour_list[ind_:ind_+self.batch_size]
            next_board_tensor, my_next_move_layer, terminals, pred_moves, pred_proms, pred_values, best_opponent_move = self.batched_child_move(
                index_boards, index_actions, index_promotions, index_colour_list
            )
            next_boards.append(next_board_tensor)
            next_move_layers.append(my_next_move_layer)
            all_terminals.append(terminals)
            all_pred_moves.append(pred_moves)
            all_pred_proms.append(pred_proms)
            all_pred_values.append(pred_values)
            best_opponent_moves.append(best_opponent_move)

        next_boards = torch.cat(next_boards, dim=0)
        next_move_layers = torch.cat(next_move_layers, dim=0)
        all_terminals = torch.cat(all_terminals, dim=0)
        all_pred_moves = torch.cat(all_pred_moves, dim=0)
        all_pred_proms = torch.cat(all_pred_proms, dim=0)
        all_pred_values = torch.cat(all_pred_values, dim=0)
        best_opponent_moves = torch.cat(best_opponent_moves, dim=0)

        # Register the children for this node.
        action_index = 0
        for node in nodes:
            num_actions = node.valid_actions.shape[0]
            for act_num in range(num_actions):
                node_act_num = action_index + act_num
                state = torch.unsqueeze(next_boards[node_act_num], 0)
                move = torch.unsqueeze(next_move_layers[node_act_num], 0)
                term = all_terminals[node_act_num]
                child_node = MCTSNode(state, move, term)
                child_node.add_predictions(torch.unsqueeze(all_pred_moves[node_act_num], 0), torch.unsqueeze(all_pred_proms[node_act_num], 0), all_pred_values[node_act_num], best_opponent_moves[node_act_num])
                node.children.append(child_node)
            node.set_non_leaf()
            action_index += num_actions
    
    def fake_generate_children(self, node):
        node.is_leaf = False
    
    @conditional_compile
    def get_max_ucbi_index(self, values_tensor):
        # UCB1
        ucbi_tensor = values_tensor[:,0] + 2 * torch.sqrt(torch.log(values_tensor[:,1]) / values_tensor[:,2])
        max_index = torch.argmax(ucbi_tensor)
        return max_index        
    
    def get_max_ucbi_node(self, current_state):
        for child in current_state.children:
            if child.N == 0:  # Shortcut - speeds things up slightly.
                return child
        values_tensor = torch.tensor([[child.Q, current_state.N, child.N] for child in current_state.children]).cuda()
        max_index = self.get_max_ucbi_index(values_tensor)
        return current_state.children[max_index]

    def generate_graph(self):
        # Compilation time is 2 minutes or longer, so disable this if debugging.
        # Takes ~20s for move 1 if uncompiled and ~12s if compiled, currently. Look into ways to make this faster.
        self.boards.update_batch_size(self.batch_size)
        nodes_marked_for_generation = []
        nodes_marked_for_rollout = []
        for _ in range(0, self.depth):
            current_state = self.top_node
            # Find the next node to roll out and then do so.
            backup_states = [current_state]
            while not current_state.is_leaf:
                # Generate children for multiple nodes at once to minimise the number of separate GPU calls
                if current_state.marked_for_generation:
                    # Rollout for multiple nodes at once to minimise the number of separate GPU calls
                    if current_state != self.top_node:  # Check if top node can also be rolled out here
                        for n in nodes_marked_for_rollout:
                            n.marked_for_rollout = False
                        self.rollout(nodes_marked_for_rollout)
                        nodes_marked_for_rollout.clear()

                    for generation_node in nodes_marked_for_generation:
                        generation_node.marked_for_generation = False
                    self.generate_children(nodes_marked_for_generation)
                    nodes_marked_for_generation.clear()

                current_state = self.get_max_ucbi_node(current_state)
                backup_states.append(current_state)

            # Mark any eligible states for rollout
            if not current_state.rollout_done and not current_state.marked_for_rollout:
                nodes_marked_for_rollout.append(current_state)
                self.add_state_value(current_state)
                current_state.marked_for_rollout = True
            game_over = torch.any(current_state.terminal_status)  # Most probably a win if so.

            # If we have found a best "winning" state, keep on going to increase its N value.
            if current_state.N >= 1 and not game_over:
                current_state.marked_for_generation = True
                self.fake_generate_children(current_state)
                nodes_marked_for_generation.append(current_state)

            # Backpropagate
            value = current_state.state_value
            for back_state in backup_states:
                back_state.N = back_state.N + 1
                back_state.W = back_state.W + value
                back_state.Q = back_state.W / back_state.N

        # Clean up before these arrays go out of scope.
        for rollout_node in nodes_marked_for_rollout:
            rollout_node.marked_for_rollout = False
        self.rollout(nodes_marked_for_rollout)
        nodes_marked_for_rollout.clear()
        for generation_node in nodes_marked_for_generation:
            generation_node.marked_for_generation = False
        self.generate_children(nodes_marked_for_generation)
        nodes_marked_for_generation.clear()
        self.boards.update_batch_size(1)
    
    def choose_move_and_update_graph(self, is_training):
        child_n_values = [child.N for child in self.top_node.children]
        if is_training:
            temperature_values = [(n / self.top_node.N) ** self.top_node.temperature for n in child_n_values]
            sum_temp = sum(temperature_values)
            norm_temp = [t / sum_temp for t in temperature_values]
            choice = np.random.choice(len(norm_temp), 1, p=norm_temp)
        else:
            highest_n = max(child_n_values)
            choice = [child_n_values.index(highest_n)]
        action = self.top_node.valid_actions[choice]
        promotion = self.top_node.valid_promotions[choice]
        self.top_node = self.top_node.children[choice[0]]
        return action, promotion
    
    def update_true_last_opponent_move(self):
        # Deal with white MCTS if opponent's move was not what we thought (maybe move this to its own method in MCTSGraph)
        if self.top_node is not None:
            predicted_opponent_move = self.top_node.opponent_move_to_get_here
            actual_move = self.top_node.opponent_actual_move
            if predicted_opponent_move is not None and not torch.equal(actual_move, predicted_opponent_move):
                self.reset_graph()  # Chuck out all our computation and start over.


class A2CGameMemory:
    # Works very similarly to DQNExperienceBuffer
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.state_buffer = torch.empty((0, 8, 8, 8)).to(torch.int8)
        self.mcts_prob_buffer = []  # Variable size
        self.game_value_buffer = torch.empty((0)).to(torch.float32)
        self.game_num_buffer = torch.empty((0)).to(torch.int32)  # Mostly for reference
        self.training_batch_size = batch_size

    def add_to_memory(self, state, mcts_probs, game_val, game_num):
        if self.state_buffer.shape[0] > self.max_size:
            begin = self.state_buffer.shape[0] - self.max_size
        else:
            begin = 0
        game_vals = torch.ones((state.shape[0])).to(torch.float32) * game_val
        game_nums = torch.ones((state.shape[0])).to(torch.int32) * game_num
        self.state_buffer = torch.cat((self.state_buffer[begin:], state), dim=0)
        self.game_value_buffer = torch.cat((self.game_value_buffer[begin:], game_vals), dim=0)
        self.game_num_buffer = torch.cat((self.game_num_buffer[begin:], game_nums), dim=0)
        for del_index in range(begin):
            self.mcts_prob_buffer.pop(0)
        self.mcts_prob_buffer = self.mcts_prob_buffer + mcts_probs

    def sample_training_batch(self):
        rand_sample_indices = torch.randint(low=0, high=self.state_buffer.shape[0], size=(self.training_batch_size,))
        states = self.state_buffer[rand_sample_indices].cuda()
        mcts_probs = [self.mcts_prob_buffer[ind].cuda() for ind in rand_sample_indices]
        game_vals = self.game_value_buffer[rand_sample_indices].cuda()
        return states, mcts_probs, game_vals


class A2CMoveAgent(JLEAIMoveAgent):
    """
    An agent that will use the A3C algorithm with a single processor, i.e. without
    the "asynchronous" part, turning it into the simpler A2C algorithm.
    """

    def __init__(self, boards, starting_position, enabled_optional_rewards, artifacts_dir):
        self.boards = boards
        self.starting_position = starting_position
        self.running_white_states = []
        self.running_white_prob = []
        self.running_black_states = []
        self.running_black_prob = []
        self.model = A2CChessNetwork()
        self.training = True
        self.whites_move = True  # True is white, False is black. We can reason this way with a batch size of 1.
        self.win_reward = 100.0 / 100.0 # 100
        self.lose_reward = -100 / 100.0 # -100
        self.move_reward = -1 / 100.0
        self.max_episode_length = 64
        self.wins = 0
        self.losses = 0
        self.white_mcts = MCTSGraph(self, boards)
        self.black_mcts = MCTSGraph(self, boards)
        now_str = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.artifacts_dir = Path('.') / artifacts_dir / now_str
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.training_memory = A2CGameMemory(10_000, 256)
        self.num_iter_delta = 100
        self.num_iters_to_train = 100

    def end_episode(self):
        self.whites_move = True
        self.running_white_prob.clear()
        self.running_white_states.clear()
        self.running_black_prob.clear()
        self.running_black_states.clear()
        # torch.cuda.empty_cache()  # Dodgy - nvidia-smi is all over the place with this.

    def start_episode(self):
        self.white_mcts.reset_graph()
        self.black_mcts.reset_graph()

    def _decide_move_for_player(self, board_tensor, player_mcts, player_states, player_probs):
        # Get the moves
        move_layer = chess_cpp.get_moves_for_player(board_tensor)
        player_mcts.update_true_last_opponent_move()  # Potentially resets the graph if opponent did something different to our assumption.
        player_mcts.init_top_node_if_empty_graph(board_tensor, move_layer)
        start = time.time()
        with torch.no_grad():
            self.model.set_test_mode()
            if torch.sum(move_layer) > 0:
                player_mcts.generate_graph()  # The new way.
                print(time.time() - start)
                if self.training:
                    player_states.append(player_mcts.top_node.current_board.clone().cpu())
                    player_probs.append(player_mcts.top_node.get_probability_distribution().cpu())
            self.model.set_train_mode()
        if torch.sum(move_layer) > 0:
            a2c_move, a2c_promotion = player_mcts.choose_move_and_update_graph(is_training=self.training)
        else:
            raise ValueError("No nodes to choose from.")
        return a2c_move, a2c_promotion

    def _inform_other_player_of_move(self, other_mcts, a2c_move):
        actual_move = torch.squeeze(a2c_move, 0)
        if other_mcts.top_node is not None:
            other_mcts.top_node.set_opponent_actual_move(actual_move)

    def decide_move(self, board_state):
        board_tensor, _, _ = board_state
        # Choose a move
        if self.whites_move:
            a2c_move, a2c_promotion = self._decide_move_for_player(board_tensor, self.white_mcts, self.running_white_states, self.running_white_prob)
            self._inform_other_player_of_move(self.black_mcts, a2c_move)
            # Next to do 17/04/2026:
            # Speed up generating graph even more, think of ways
            # Fix any errors
            # Fix memory leak - GPU usage creeps up as training goes on
            # Handling of draws by threefold repetition and 50 move rule will need to be decided with a meta-layer, since they are ignored during exploration.
            # Test games with white and black vs. Stockfish
            # Experiment with multithreading
        else:
            a2c_move, a2c_promotion = self._decide_move_for_player(board_tensor, self.black_mcts, self.running_black_states, self.running_black_prob)
            self._inform_other_player_of_move(self.white_mcts, a2c_move)
        self.boards.update_batch_size(1)  # To be safe.
        return (a2c_move, a2c_promotion)

    def prepare_for_training(self):
        self.training = True
        self.model.set_train_mode()

    def prepare_for_evaluation(self):
        self.training = False
        self.model.set_test_mode()

    def save_all_models(self):
        self.model.save_models()

    def load_all_models(self):
        self.model.load_models('train')

    def save_game_to_memory(self, true_game_value, game_num):
        w_states_this_game = torch.cat(self.running_white_states, dim=0)
        b_states_this_game = torch.cat(self.running_black_states, dim=0)
        self.training_memory.add_to_memory(w_states_this_game, self.running_white_prob, true_game_value.cpu(), game_num)
        self.training_memory.add_to_memory(b_states_this_game, self.running_black_prob, -true_game_value.cpu(), game_num)

    def train_step(self, epoch):
        states, mcts_probs, game_vals = self.training_memory.sample_training_batch()
        self.model.update_network(states, mcts_probs, game_vals)

    def train_on_data(self, start_epoch):
        print(f"Training for {self.num_iters_to_train} iterations...")
        for current_epoch in range(start_epoch, start_epoch + self.num_iters_to_train):
            self.train_step(current_epoch)
        return current_epoch

    def update_training_params(self, num_logged_games):
        self.num_iters_to_train += self.num_iter_delta

    def log_opponent_move(self, move, promotion, board_state):
        # A generic component of log_stockfish_move
        actual_move = torch.squeeze(move, 0)
        if self.whites_move:
            self.white_mcts.top_node.set_opponent_actual_move(actual_move)
        else:
            self.black_mcts.top_node.set_opponent_actual_move(actual_move)

    def log_stockfish_move(self, move, board_state, starting_colour_me):
        jle_move, jle_promotion = convert_UCI_to_jle_notation(move, starting_colour_me)
        self.log_opponent_move(jle_move, jle_promotion, board_state)
        return self.enact_move((jle_move, jle_promotion), board_state)

    def self_play_and_training_session(self, boards, start_epoch):
        """
        Plays a game against itself (a full episode) and then learns on the data.
        """
        boards.update_batch_size(1)
        batched_board = boards.to_tensor().cuda()
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((1)).to(torch.bool).cuda()
        current_epoch = start_epoch
        
        self.wins = 0
        self.losses = 0
        num_logged_games = 0
        total_desired_logged_games = 100
        train_cadence = 5  # After how many more logged games to do a training cycle
        total_games = 0
        while num_logged_games < total_desired_logged_games:
            states = [batched_board.clone()]
            moves = []
            promotions = []
            game_over = False
            episode_length = 0
            self.start_episode()
            game_length = 1

            while not game_over:
                (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
                    self.decide_and_enact_move((batched_board, colour_list, dud_move_count))
                states.append(batched_board.clone())
                moves.append(move)
                promotions.append(promotion)
                game_over = torch.any(game_over_tensor, dim=1)
                self.whites_move = not self.whites_move
                episode_length += 1
                if episode_length >= self.max_episode_length:
                    batched_board, colour_list, dud_move_count = self.reset_all((batched_board, colour_list, dud_move_count))
                    break
                game_length = game_length + 1

            game_over_message = get_game_over_message(game_over_tensor[0], colour_list[0])
            if game_over_message is None:
                game_over_message = "Max game length reached. Terminating game"
            print(f"{game_over_message}, game length: {game_length}")
            episode_length = 0
            major_outcomes = game_over_tensor[0, 0:3]
            if torch.any(major_outcomes):
                true_game_value = get_game_value_for_white(major_outcomes, colour_list[0]).detach()
                self.save_game_to_memory(true_game_value, num_logged_games)
                num_logged_games += 1
                if num_logged_games % train_cadence == 0:
                    self.model.set_train_mode()
                    current_epoch = self.train_on_data(current_epoch)
                    self.update_training_params(num_logged_games)
                    self.model.set_test_mode()
            else:
                self.model.optimiser.zero_grad()
            save_full_game_artifacts(self.artifacts_dir, total_games + 1, states, moves, promotions)
            self.end_episode()
            total_games += 1
            print(total_games)
        print(f"Wins {self.wins}, Losses {self.losses}")
        
        return current_epoch


class RandomMoveAgent(JLEAIMoveAgent):
    """
    An agent that picks a move randomly.
    """

    def __init__(self, boards, starting_position):
        self.boards = boards
        self.starting_position = starting_position
    
    @conditional_compile
    def decide_and_enact_move(self, board_state):
        return super().decide_and_enact_move(board_state)
    
    def prepare_for_training(self):
        pass
    
    def prepare_for_evaluation(self):
        pass

    def self_play_and_training_session(self, boards, start_epoch):
        pass

    def train_step(self, epoch):
        pass

    @conditional_compile
    def decide_move(self, board_state):
        board_tensor, _, _ = board_state
        # Theoretically, the line below is about 1 second faster over 10,000 moves, but more complicated.
        # move_layer = get_moves_for_player_with_reuse(batched_board, game_over, opponent_move_layer)
        # Get the moves
        move_layer = chess_cpp.get_moves_for_player(board_tensor)
        # Choose a move
        randomly_selected_move, random_promotion = get_random_move(board_tensor, move_layer)
        return (randomly_selected_move, random_promotion)
    
    def save_all_models(self):
        pass
    
    def load_all_models(self):
        pass


def evaluate_against_stockfish(boards, stockfish_agent, our_ai_agent):
    """
    Evaluate one of the AIs against Stockfish at some difficulty level.
    """
    boards.update_batch_size(1)

    stockfish_agent.set_elo(0)  # Min ELO is now 1,320!!
    for i in range(0, 20):
        stockfish_agent.start_new_game()
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((1)).to(torch.bool).cuda()
        single_board = boards.to_tensor().cuda()
        winner = ''
        if random() > 0.5:  # 0.5
            # Play white.
            starting_colour = True
            (randomly_selected_move, random_promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            stockfish_agent.log_opponent_move(randomly_selected_move, random_promotion, single_board, starting_colour)
        else:
            # Play black.
            starting_colour = False
        while True:
            stockfish_move, _ = stockfish_agent.decide_and_enact_move(single_board)
            (jle_move, jle_promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.log_stockfish_move(stockfish_move, (single_board, colour_list, dud_move_count), starting_colour)
            
            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'Stockfish'
                else:
                    winner = 'Draw'
                break

            (randomly_selected_move, random_promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            flipped_board, _ = flip_board(single_board, colour_list) #####
            stockfish_agent.log_opponent_move(randomly_selected_move, random_promotion, flipped_board, starting_colour)
            
            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'JLE'
                else:
                    winner = 'Draw'
                break
    
        print(f"I did a whole game - The winner was {winner}")

    boards.update_batch_size(BATCH_SIZE)


def evaluate_a2c_against_random(boards, random_agent, our_ai_agent):
    """
    Evaluate an A2C AI against random, to test whether training is working.
    """
    boards.update_batch_size(1)

    for i in range(0, 20):
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((1)).to(torch.bool).cuda()
        single_board = boards.to_tensor().cuda()
        winner = ''
        episode_length = 0
        our_ai_agent.start_episode()
        inform_about_opponent_move = True  # When we play black, do not inform about the first move white takes.
        # This is due to starting the graph at our first playable move, which will not be the usual chess starting state.
        #col = False
        if random() > 0.5:  # 0.5
            # Play white.
            #col = True
            (move, promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            episode_length += 1
        else:
            our_ai_agent.whites_move = False
            inform_about_opponent_move = False
        while True:
            #col = not col
            (move, promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                random_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            if inform_about_opponent_move:
                our_ai_agent.log_opponent_move(move, promotion, (single_board, colour_list, dud_move_count))
            else:
                inform_about_opponent_move = True
            episode_length += 1
            if episode_length >= our_ai_agent.max_episode_length:
                winner = 'no winner. The game went on too long.'
                break

            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'Random'
                else:
                    winner = 'Draw'
                break

            #col = not col
            (move, promotion), (dud_move_count, single_board, colour_list, _, game_over_tensor) = \
                our_ai_agent.decide_and_enact_move((single_board, colour_list, dud_move_count))
            episode_length += 1
            if episode_length >= our_ai_agent.max_episode_length:
                winner = 'no winner. The game went on too long.'
                break

            if torch.any(game_over_tensor):
                if game_over_tensor[0][0]:
                    winner = 'JLE'
                else:
                    winner = 'Draw'
                break

        single_board, colour_list, dud_move_count = our_ai_agent.reset_all((single_board, colour_list, dud_move_count))
        our_ai_agent.end_episode()
        print(f"I did a whole game - The winner was {winner}")

    boards.update_batch_size(BATCH_SIZE)


@conditional_compile
def evaluate_dqn_against_random(boards, random_agent, our_ai_agent):
    """
    Play an DQN AI against random, to test whether a new training algorithm is working.
    """
    boards.update_batch_size(256)
    batched_board = boards.to_tensor().cuda()
    dud_move_count = boards.get_starting_move_count_list()
    colour_list = torch.ones((batched_board.shape[0])).to(torch.bool).cuda()
    win_count = 0
    lose_count = 0
    draw_count = 0
    games_played = 0
    while games_played < 1000:
        # The trained AI always starts white, for the sake of argument.
        (move, promotion), (dud_move_count, batched_board, colour_list, _, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        games_won = torch.sum(game_over_tensor[:,0].to(torch.int8))
        games_drawn = torch.sum(game_over_tensor[:,1:].to(torch.int8))
        win_count += games_won
        draw_count += games_drawn
        games_played += (games_won + games_drawn)
        
        (move, promotion), (dud_move_count, batched_board, colour_list, _, game_over_tensor) = \
            random_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        games_lost = torch.sum(game_over_tensor[:,0].to(torch.int8))
        games_drawn = torch.sum(game_over_tensor[:,1:].to(torch.int8))
        lose_count += games_lost
        draw_count += games_drawn
        games_played += (games_lost + games_drawn)
        #print(games_played)
    print(f"Win count: {win_count} ({win_count / games_played * 100}%), "
          f"Lose count: {lose_count} ({lose_count / games_played * 100}%), "
          f"Draw count: {draw_count} ({draw_count / games_played * 100}%), ")


def get_args():
    parser = argparse.ArgumentParser(description='Parameters for training a chess engine.')
    parser.add_argument(
        'algorithm',
        type=str,
        default="DQN",
        choices=["DQN", "A2C"],
        help='The algorithm to use for training.'
    )
    parser.add_argument(
        '--artifacts_dir',
        type=str,
        default='artifacts',
        help='The folder to save artifacts in, starting from the present directory.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Starting condition
    args = get_args()
    # Insist that we have CUDA for now, otherwise things will be much slower.
    assert torch.cuda.is_available(), "CUDA is not enabled. Please fix this before running this script."
    torch._dynamo.config.cache_size_limit = 64
    mode = 1
    mode_str = "full" if mode == 0 else "simplified"
    boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, mode)
    batched_board = boards.to_tensor().cuda()
    starting_position = batched_board[0].clone()
    stockfish_agent = StockfishMoveAgent(boards, starting_position)
    print(f"Training with {mode_str} chess games")
    if args.algorithm == "DQN":
        our_ai_agent = DQNMoveAgent(boards, starting_position, {})  # {"firepower", "firepower_per_num_moves"}
        # Recommended batch size = 256
    elif args.algorithm == "A2C":
        our_ai_agent = A2CMoveAgent(boards, starting_position, {}, args.artifacts_dir)
        assert boards.get_batch_size() == 1
    #our_ai_agent.load_all_models()
    random_agent = RandomMoveAgent(boards, starting_position)
    our_ai_agent.prepare_for_training()
    current_epoch = 0
    #current_epoch = our_ai_agent.self_play_and_training_session(boards, current_epoch)
    """for i in range(0, 1):
        our_ai_agent.prepare_for_training()
        current_epoch = our_ai_agent.self_play_and_training_session(boards, current_epoch)"""
        #our_ai_agent.prepare_for_evaluation()
        #evaluate_dqn_against_random(boards, random_agent, our_ai_agent)
    our_ai_agent.prepare_for_evaluation()
    evaluate_a2c_against_random(boards, random_agent, our_ai_agent)
    #evaluate_against_stockfish(boards, stockfish_agent, our_ai_agent)
    #our_ai_agent.save_all_models()
