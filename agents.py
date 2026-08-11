# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:30:59 2026

@author: jledragon
"""

import time
import torch
import chess_cpp
from stockfish import Stockfish
import platform
import numpy as np
from interface import AIMoveAgent, JLEAIMoveAgent
from constants import BATCH_SIZE, TOTAL_DESIRED_LOGGED_GAMES_A2C, A2C_TRAIN_CADENCE, MCTS_BATCH_SIZE
import torch.multiprocessing as mp
from neural_networks import A2CChessNetwork
from eval_util import evaluate_agent_against_random
from chess_py_utils import (
    get_random_move,
    convert_jle_to_UCI_notation,
    convert_UCI_to_jle_notation,
    get_human_readable_board,
    flip_episode,
    get_firepower_score,
    conditional_compile,
    get_game_over_message,
    get_game_value_for_white,
    save_full_game_artifacts,
    flip_states,
    #get_moves_for_player_with_reuse
)


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
        self.stockfish.make_moves_from_start()
    
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
        self.stockfish.make_moves_from_current_position([move])
        return move, board_state
    
    def log_opponent_move(self, move, promotion, board_tensor, starting_colour_opponent):
        """
        Log the move of the opponent AI to Stockfish.
        """
        uci_move = convert_jle_to_UCI_notation(move, promotion, board_tensor, not starting_colour_opponent)
        try:
            self.stockfish.make_moves_from_current_position([uci_move])
        except Exception as e:
            print("Stockfish's version of the game is:\n")
            print(self.stockfish.get_board_visual())
            print("JLE version of the game is:\n")
            print(get_human_readable_board(board_tensor[0], not starting_colour_opponent))
            print("Moves taken to get here:\n")
            raise e


class DQNExperienceBuffer:
    def __init__(self, max_size, batch_size, use_state_actions):
        self.max_size = max_size
        self.state_buffer = torch.empty((0, 6, 8, 8)).to(torch.int8)
        self.move_buffer = torch.empty((0, 4)).to(torch.int8)
        self.promotion_buffer = torch.empty((0, 4)).to(torch.int8)
        self.rewards_buffer = torch.empty((0)).to(torch.int8)
        self.terminals_buffer = torch.empty((0)).to(torch.int8)
        self.next_state_buffer = torch.empty((0, 6, 8, 8)).to(torch.int8)
        self.next_move_buffer = torch.empty((0, 4)).to(torch.int8)
        self.next_promotion_buffer = torch.empty((0, 4)).to(torch.int8)
        self.training_batch_size = batch_size
        self.use_state_actions = use_state_actions

    def add_to_buffer(self, state, action, rewards, terminals, next_state, next_action):
        move, promotion = action
        next_move, next_promotion = next_action
        if self.state_buffer.shape[0] > self.max_size:
            begin = state.shape[0]
        else:
            begin = 0
        self.state_buffer = torch.cat((self.state_buffer[begin:], state), dim=0)
        self.move_buffer = torch.cat((self.move_buffer[begin:], move), dim=0)
        self.promotion_buffer = torch.cat((self.promotion_buffer[begin:], promotion), dim=0)
        self.rewards_buffer = torch.cat((self.rewards_buffer[begin:], rewards), dim=0)
        self.terminals_buffer = torch.cat((self.terminals_buffer[begin:], terminals), dim=0)
        self.next_state_buffer = torch.cat((self.next_state_buffer[begin:], next_state), dim=0)
        if self.use_state_actions:
            self.next_move_buffer = torch.cat((self.next_move_buffer[begin:], next_move), dim=0)
            self.next_promotion_buffer = torch.cat((self.next_promotion_buffer[begin:], next_promotion), dim=0)

    def sample_training_batch(self):
        rand_sample_indices = torch.randint(low=0, high=self.state_buffer.shape[0], size=(self.training_batch_size,))
        states = self.state_buffer[rand_sample_indices].cuda()
        moves = self.move_buffer[rand_sample_indices].cuda()
        promotions = self.promotion_buffer[rand_sample_indices].cuda()
        rewards = self.rewards_buffer[rand_sample_indices].cuda()
        terminals = self.terminals_buffer[rand_sample_indices].cuda()
        next_states = self.next_state_buffer[rand_sample_indices].cuda()
        if self.use_state_actions:
            next_moves = self.next_move_buffer[rand_sample_indices].cuda()
            next_promotions = self.next_promotion_buffer[rand_sample_indices].cuda()
            next_action = (next_moves, next_promotions)
        else:
            next_action = None
        return states, (moves, promotions), rewards, terminals, next_states, next_action

    def save_data(self):
        pass  # TODO

    def load_data(self):
        pass  # TODO


class DQNMoveAgent(JLEAIMoveAgent):
    """
    A neural network, reinforcement learning-based algorithm to choose moves.
    """

    def __init__(self, boards, model, memory, starting_position, enabled_optional_rewards, use_state_actions):
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
        self.experience_buffer = memory
        self.q_network = model
        self.win_reward = 1 # 100
        self.lose_reward = -1 # -100
        self.move_reward = -0.05 # -1
        self.update_rate = 40 # 40
        self.move_reward_factor = 6 # 6
        self.enabled_optional_rewards = enabled_optional_rewards
        self.use_state_actions = use_state_actions

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
        # Assumption - the game cannot be over in one move. If it could, and this were the first
        # move in the next game, then this would interfere with the previous game.
        board_state, opponent_move_layer = game_state_bundle
        if self.opponent_rewards is not None:
            losses_opp = game_over_tensor[:, 0].cpu()
            other_game_over_opp_turn = torch.any(game_over_tensor[:, 1:], dim=1).cpu()
            # ^ Yes, even non-positional draws...
            num_moves_reward = -self._move_reward_fn(opponent_move_layer)
            punishment = num_moves_reward.cpu()
            self.opponent_rewards += punishment
            if torch.sum(losses_opp) > 0:
                self.opponent_terminals[losses_opp] = True
                self.opponent_rewards[losses_opp] = self.opponent_rewards[losses_opp] + self.lose_reward
            elif torch.sum(other_game_over_opp_turn) > 0:
                self.opponent_terminals[other_game_over_opp_turn] = True
    
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
            self.experience_buffer.add_to_buffer(self.previous_state, self.previous_action, self.previous_rewards, self.previous_terminals, current_state[:,:6,:,:].cpu(), (move.cpu(), promotion.cpu()))
        # Store state, actions, rewards and terminals for the next pass
        self.store_current_artifacts(current_state[:,:6,:,:].cpu(), (move.cpu(), promotion.cpu()), (rewards.cpu()), game_over.cpu())
    
    def train_step(self, epoch, augment_data=True):
        for update in range(0, self.update_rate):
            states, actions, rewards, terminals, next_states, next_actions = self.experience_buffer.sample_training_batch()
            if augment_data:
                # Doubles our data by flipping horizontally, which is always invariant for this chess setup.
                states, actions, next_states, next_actions = flip_episode(states, actions, next_states, next_actions)
            # TODO - randomly augment data by flipping horizontally
            self.q_network.update_network(states, actions, rewards, terminals, next_states, next_actions)
            self.q_network.soft_target_update()
        # Training strategy here
        if self.q_network.eps > 0.05:
            self.q_network.eps = self.q_network.eps * 0.998 # 0.9942
        else:
            self.q_network.eps = 0.05
        if epoch == 2100:
            for group in self.q_network.optimiser.param_groups:
                group['lr'] = group['lr'] * 0.1

    def self_play_and_training_session(self, boards, dud_move_count, colour_list, start_epoch, total_games):
        """
        Play an AI against itself and train on past data.
        """
        batched_board = boards.to_tensor().cuda()
        # Run evaluation against random. Gives us an indicator of how training is going - future: Make this configurable.
        starting_position = batched_board[0].clone()
        random_agent = RandomMoveAgent(boards, starting_position)
        evaluate_agent_against_random(boards, random_agent, self)

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
                self.train_step(move_num, augment_data=False)
            games_done = torch.sum(game_over.to(torch.int8))
            total_games += games_done
        print(self.q_network.eps, start_epoch, total_games.cpu().item())
        return boards, dud_move_count, colour_list, end_epoch, total_games

    def decide_move(self, board_state):
        board_tensor, _, _ = board_state
        # Get the moves
        move_layer = chess_cpp.get_moves_for_player(board_tensor)
        # Choose a move
        if self.use_state_actions:
            dqn_move, dqn_promotion = self.q_network.get_move_state_action(board_tensor[:,:6,:,:], move_layer)
        else:
            dqn_move, dqn_promotion = self.q_network.get_move_state_only(board_tensor[:,:6,:,:], move_layer)
        return (dqn_move, dqn_promotion)
    
    def load_all_models(self):
        self.q_network.load_models('train')


def train_dqn(args, model, memory, board_setup, use_state_actions):
    boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, board_setup)
    assert boards.get_batch_size() > 32
    # Recommended batch size = 256, min. 32.
    batched_board = boards.to_tensor().cuda()
    starting_position = batched_board[0].clone()
    our_ai_agent = DQNMoveAgent(boards, model, memory, starting_position, {}, use_state_actions)  # {"firepower", "firepower_per_num_moves"}
    our_ai_agent.prepare_for_training()
    current_epoch = 0
    total_games = 0

    dud_move_count = boards.get_starting_move_count_list()
    colour_list = torch.ones((BATCH_SIZE)).to(torch.bool).cuda()
    for session in range(0, 50):
        print(f"Starting new batch of 100 moves from {current_epoch}.")
        boards, dud_move_count, colour_list, current_epoch, total_games = our_ai_agent.self_play_and_training_session(
            boards, dud_move_count, colour_list, current_epoch, total_games
        )


class MCTSNode:
    """
    A single node in MCTS representing a state and its attributes for MCTS.
    """
    
    def __init__(self, current_state, moves_for_state, terminal_status, colour):
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
        self.temperature = 2
        self.marked_for_generation = False  # An attempt to make graph generation faster.
        self.action_probs_with_grad = None
        self.state_value = None  # Information about predicted value or better - may be GT terminal
        self.marked_for_rollout = False
        self.colour = colour
    
    def update_actions(self, valid_actions, valid_promotions, action_probs):
        self.P = action_probs.detach().cpu().numpy()
        self.valid_actions = valid_actions
        self.valid_promotions = valid_promotions

    def update_values(self, state_value, predicted_value):
        self.state_value = state_value.detach().cpu().numpy()
        self.predicted_value = predicted_value
    
    def add_predictions(self, model_move, model_promotion, pred_value):
        # Add the predictions that led us to this state. This saves us a lot of computation time by batching up model predictions.
        self.model_move = model_move
        self.model_promotion = model_promotion
        self.pred_value = pred_value
    
    def set_non_leaf(self):
        self.is_leaf = False
        #self.current_board = None
        self.moves_for_state = None
        self.model_move = None
        self.pred_value = None
    
    def get_probability_distribution(self):
        child_ns = [child.N for child in self.children]
        temperature_values = [(n / self.N) ** self.temperature for n in child_ns]
        sum_temp = sum(temperature_values)
        norm_temp = torch.Tensor([t / sum_temp for t in temperature_values]).cuda().to(torch.float32)
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
        self.value_mask = torch.tensor([1, 0]).to(torch.float32).cuda()
        self.batch_size = MCTS_BATCH_SIZE
        self.nodes_marked_for_generation = []
        self.nodes_marked_for_rollout = []
        self.cpuct_base = 19_652
        self.cpuct_init = 2.5
        self.mcts_noise = 0.25
        self.dir_alpha = 0.3

    def reset_graph(self):
        self.top_node = None
    
    #@conditional_compile
    def init_top_node_if_empty_graph(self, current_board, moves_for_state):
        if self.top_node is None:
            not_terminal = torch.zeros((2)).to(torch.bool).cuda()  # No win, no draw
            self.top_node = MCTSNode(current_board, moves_for_state, not_terminal, True)

            # Special - only do this for the first move of the game.
            self.top_node.rollout_done = True
            if torch.sum(moves_for_state) > 0:
                # Possible no grad here.
                model_moves, model_proms, state_value = self.model.get_model_move_and_state(current_board[:,:6,:,:])
                self.top_node.add_predictions(None, None, state_value)
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
        scale = 1 if node.colour else -1
        state_value = self.value_mask[node.terminal_status] * scale
        state_value = node.pred_value if state_value.shape[0] == 0 else state_value
        assert not state_value.shape[0] > 1, f"Error - node end state had more than one outcome. Terminal status for node: {node.terminal_status}"
        node.update_values(state_value, node.pred_value)

    #@conditional_compile
    def batched_child_move(self, index_boards, index_actions, index_promotions, index_colour_list, index_node_colours):
        (_, _), (_, board_tensor_half, index_colour_list, opponent_move_layer, game_over_tensor_1) = \
            self.agent.enact_move((index_actions, index_promotions), (index_boards, index_colour_list, None), full_game_over_conditions=False)
        we_won = torch.unsqueeze(game_over_tensor_1[:, 0], 1)
        we_drew = torch.unsqueeze(torch.any(game_over_tensor_1[:, 1:3], axis=1), 1)  # Only account for stalemates and too few pieces as draws

        terminals = torch.cat((we_won, we_drew), axis=1)
        pred_moves, pred_proms, pred_values = self.model.get_model_move_and_state(board_tensor_half[:,:6,:,:])
        pred_values = pred_values * index_node_colours
        return board_tensor_half, opponent_move_layer, terminals, pred_moves, pred_proms, pred_values

    #@conditional_compile  # Somehow slows things down?
    def generate_children(self, nodes):
        if len(nodes) == 0:
            return
        total_num_actions = 0
        all_current_boards = []
        all_valid_actions = []
        all_promotions = []
        all_colour_list = []
        all_node_colours = []
        for node in nodes:
            current_board = node.current_board
            valid_actions = node.valid_actions
            all_valid_actions.append(valid_actions)
            all_promotions.append(node.valid_promotions)
            num_actions = valid_actions.shape[0]
            total_num_actions += num_actions
            all_node_colours.append(torch.tensor(1 if node.colour else -1).cuda().unsqueeze(0).repeat(num_actions, 1))

            # Enact each action on copies of the board.
            multi_board = current_board.repeat(num_actions, 1, 1, 1)
            colour_list = torch.ones((num_actions)).to(torch.bool).cuda()  # We are always us, or "white" when considering expansion.
            all_current_boards.append(multi_board)
            all_colour_list.append(colour_list)
        all_current_boards = torch.cat(all_current_boards, dim=0)
        all_valid_actions = torch.cat(all_valid_actions, dim=0)
        all_promotions = torch.cat(all_promotions, dim=0)
        all_colour_list = torch.cat(all_colour_list, dim=0)
        all_node_colours = torch.cat(all_node_colours, dim=0)
        next_boards = []
        next_move_layers = []
        all_terminals = []
        all_pred_moves = []
        all_pred_proms = []
        all_pred_values = []
        # Keep the child node computation up to some maximum batch size
        for ind_ in range(0, total_num_actions, self.batch_size):
            index_boards = all_current_boards[ind_:ind_+self.batch_size,:,:,:]
            index_actions = all_valid_actions[ind_:ind_+self.batch_size,:]
            index_promotions = all_promotions[ind_:ind_+self.batch_size,:]
            index_colour_list = all_colour_list[ind_:ind_+self.batch_size]
            index_node_colours = all_node_colours[ind_:ind_+self.batch_size]
            next_board_tensor, my_next_move_layer, terminals, pred_moves, pred_proms, pred_values = self.batched_child_move(
                index_boards, index_actions, index_promotions, index_colour_list, index_node_colours
            )
            next_boards.append(next_board_tensor)
            next_move_layers.append(my_next_move_layer)
            all_terminals.append(terminals)
            all_pred_moves.append(pred_moves)
            all_pred_proms.append(pred_proms)
            all_pred_values.append(pred_values)

        next_boards = torch.cat(next_boards, dim=0)
        next_move_layers = torch.cat(next_move_layers, dim=0)
        all_terminals = torch.cat(all_terminals, dim=0)
        all_pred_moves = torch.cat(all_pred_moves, dim=0)
        all_pred_proms = torch.cat(all_pred_proms, dim=0)
        all_pred_values = torch.cat(all_pred_values, dim=0)

        # Register the children for this node.
        action_index = 0
        for node in nodes:
            num_actions = node.valid_actions.shape[0]
            for act_num in range(num_actions):
                node_act_num = action_index + act_num
                state = torch.unsqueeze(next_boards[node_act_num], 0)
                move = torch.unsqueeze(next_move_layers[node_act_num], 0)
                term = all_terminals[node_act_num]
                child_node = MCTSNode(state, move, term, not node.colour)
                child_node.add_predictions(torch.unsqueeze(all_pred_moves[node_act_num], 0), torch.unsqueeze(all_pred_proms[node_act_num], 0), all_pred_values[node_act_num])
                node.children.append(child_node)
            node.set_non_leaf()
            action_index += num_actions
    
    def fake_generate_children(self, node):
        node.is_leaf = False
    
    def get_max_puct_index(self, values_array):
        # PUCT
        cpuct = np.log((values_array[:,3] + self.cpuct_base + 1) / self.cpuct_base) + self.cpuct_init
        puct_array = (values_array[:,4] * values_array[:,0]) + cpuct * (values_array[:,2] * np.sqrt(values_array[:,1]) / (1 + values_array[:,3]))
        max_index = np.argmax(puct_array)
        return max_index

    def get_max_puct_node(self, current_state, is_training):
        #for child in current_state.children:
        #    if child.N == 0:  # Shortcut - speeds things up slightly.
        #        return child
        if is_training:
            noise = np.random.dirichlet((self.dir_alpha,) * len(current_state.P)).transpose()
            p = (1 - self.mcts_noise) * current_state.P + self.mcts_noise * noise
        else:
            p = current_state.P
        values_array = np.array([[
            child.Q,
            current_state.N,
            p[i],
            child.N,
            1 if child.colour else -1
        ] for i, child in enumerate(current_state.children)])
        max_index = self.get_max_puct_index(values_array)
        return current_state.children[max_index]

    def generate_graph(self, is_training):
        # Compilation time is 2 minutes or longer, so disable this if debugging.
        # Takes ~20s for move 1 if uncompiled and ~12s if compiled, currently. Look into ways to make this faster.
        self.boards.update_batch_size(self.batch_size)
        for _ in range(0, self.depth):
            current_state = self.top_node
            # Find the next node to roll out and then do so.
            backup_states = [current_state]
            while not current_state.is_leaf:
                # Generate children for multiple nodes at once to minimise the number of separate GPU calls
                if current_state.marked_for_generation:
                    # Rollout for multiple nodes at once to minimise the number of separate GPU calls
                    if current_state != self.top_node:  # Check if top node can also be rolled out here
                        for n in self.nodes_marked_for_rollout:
                            n.marked_for_rollout = False
                        self.rollout(self.nodes_marked_for_rollout)
                        self.nodes_marked_for_rollout.clear()

                    for generation_node in self.nodes_marked_for_generation:
                        generation_node.marked_for_generation = False
                    self.generate_children(self.nodes_marked_for_generation)
                    self.nodes_marked_for_generation.clear()

                current_state = self.get_max_puct_node(current_state, is_training)
                backup_states.append(current_state)

            # Mark any eligible states for rollout
            if not current_state.rollout_done and not current_state.marked_for_rollout:
                self.nodes_marked_for_rollout.append(current_state)
                self.add_state_value(current_state)
                current_state.marked_for_rollout = True
            game_over = torch.any(current_state.terminal_status)  # Most probably a win if so.

            # If we have found a best "winning" state, keep on going to increase its N value.
            if current_state.N >= 1 and not game_over:
                current_state.marked_for_generation = True
                self.fake_generate_children(current_state)
                self.nodes_marked_for_generation.append(current_state)

            # Backpropagate
            value = current_state.state_value
            for back_state in backup_states:
                back_state.N = back_state.N + 1
                back_state.W = back_state.W + value[0]
                back_state.Q = back_state.W / back_state.N

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


class A2CGameMemory:
    # Works very similarly to DQNExperienceBuffer
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.state_buffer = torch.empty((0, 8, 8, 8)).to(torch.int8)
        self.num_times_seen = torch.empty((0)).to(torch.int32)
        self.mcts_prob_buffer = []  # Variable size
        self.game_value_buffer = torch.empty((0)).to(torch.float32)
        self.training_batch_size = batch_size
        self.once = torch.ones((1)).to(torch.int8)
        self.augment_data = False

    def _append(self, begin, states, mcts_probs, game_val, num_times):
        game_val_template = torch.ones((states.shape[0])).to(torch.float32) * game_val
        self.state_buffer = torch.cat((self.state_buffer[begin:], states), dim=0)
        self.game_value_buffer = torch.cat((self.game_value_buffer[begin:], game_val_template), dim=0)
        self.num_times_seen = torch.cat((self.num_times_seen[begin:], num_times), dim=0)
        self.mcts_prob_buffer = self.mcts_prob_buffer + mcts_probs

    def add_to_memory(self, state, mcts_probs, game_val):
        # Delete the starting indices when memory gets too large
        if self.state_buffer.shape[0] > self.max_size:
            begin = self.state_buffer.shape[0] - self.max_size
        else:
            begin = 0
        for del_index in range(begin):
            self.mcts_prob_buffer.pop(0)
        if self.state_buffer.shape[0] == 0:
            self._append(begin, state, mcts_probs, game_val, torch.ones((state.shape[0])).to(torch.int8))
        else:
            flattened_buffer = self.state_buffer.reshape(self.state_buffer.shape[0], -1)
            for move_ind in range(0, state.shape[0]):
                state_i = state[move_ind].reshape(1, -1)
                matches = torch.all(torch.where(flattened_buffer == state_i, True, False), dim=1)
                match_ind = torch.argwhere(matches).flatten()
                if match_ind.shape[0] == 0:
                    self._append(begin, state[move_ind:move_ind+1], [mcts_probs[move_ind]], game_val, self.once)
                else:
                    mi = match_ind[0]
                    self.game_value_buffer[mi] = self.game_value_buffer[mi] + game_val
                    self.num_times_seen[mi] = self.num_times_seen[mi] + 1
                    self.mcts_prob_buffer[mi] = self.mcts_prob_buffer[mi] + mcts_probs[move_ind]

    def sample_training_batch(self):
        rand_sample_indices = torch.randint(low=0, high=self.state_buffer.shape[0], size=(self.training_batch_size,))
        states = self.state_buffer[rand_sample_indices].cuda()
        if self.augment_data:
            states = flip_states(states)
        mcts_probs = [self.mcts_prob_buffer[ind].cuda() / self.num_times_seen[ind].cuda() for ind in rand_sample_indices]
        game_vals = self.game_value_buffer[rand_sample_indices].cuda() / self.num_times_seen[rand_sample_indices].cuda()
        return states, mcts_probs, game_vals

    def save_data(self):
        data_map = {
            "state_buffer": self.state_buffer,
            "mcts_prob_buffer": self.mcts_prob_buffer,
            "game_value_buffer": self.game_value_buffer,
            "num_times_seen": self.num_times_seen,
        }
        torch.save(data_map, "datasets/latest.pt")

    def load_data(self):
        loaded = torch.load("datasets/latest.pt")
        self.state_buffer = loaded["state_buffer"]
        self.mcts_prob_buffer = loaded["mcts_prob_buffer"]
        self.game_value_buffer = loaded["game_value_buffer"]
        self.num_times_seen = loaded["num_times_seen"]

    def save_game_to_memory(self, true_game_value, running_white_states, running_black_states, running_white_prob, running_black_prob):
        w_states_this_game = torch.cat(running_white_states, dim=0)
        b_states_this_game = torch.cat(running_black_states, dim=0)
        self.add_to_memory(w_states_this_game, running_white_prob, true_game_value.cpu())
        self.add_to_memory(b_states_this_game, running_black_prob, -true_game_value.cpu())


class A2CMoveAgent(JLEAIMoveAgent):
    """
    An agent that will use the A3C algorithm with a single processor, i.e. without
    the "asynchronous" part, turning it into the simpler A2C algorithm.
    """

    def __init__(self, boards, model, starting_position, enabled_optional_rewards, use_mcts):
        self.boards = boards
        self.starting_position = starting_position
        self.running_white_states = []
        self.running_white_prob = []
        self.running_black_states = []
        self.running_black_prob = []
        self.model = model
        self.training = True
        self.whites_move = True  # True is white, False is black. We can reason this way with a batch size of 1.
        self.win_reward = 100.0 / 100.0 # 100
        self.lose_reward = -100 / 100.0 # -100
        self.move_reward = -1 / 100.0
        self.max_episode_length = 64
        self.mcts = MCTSGraph(self, boards)
        self.use_mcts = use_mcts

    def end_episode(self):
        self.whites_move = True
        self.running_white_prob.clear()
        self.running_white_states.clear()
        self.running_black_prob.clear()
        self.running_black_states.clear()
        self.mcts.nodes_marked_for_generation.clear()
        self.mcts.nodes_marked_for_rollout.clear()
        # torch.cuda.empty_cache()  # Dodgy - nvidia-smi is all over the place with this.

    def start_episode(self):
        self.mcts.reset_graph()

    def _decide_move_for_player(self, board_tensor, player_states, player_probs):
        # Get the moves
        self.model.set_test_mode()
        move_layer = chess_cpp.get_moves_for_player(board_tensor)
        self.mcts.init_top_node_if_empty_graph(board_tensor, move_layer)
        start = time.time()
        with torch.no_grad():
            if torch.sum(move_layer) > 0:
                self.mcts.generate_graph(self.training)  # The new way.
                print(time.time() - start)
                if self.training:
                    player_states.append(self.mcts.top_node.current_board.clone().cpu())
                    player_probs.append(self.mcts.top_node.get_probability_distribution().cpu())
        if torch.sum(move_layer) > 0:
            a2c_move, a2c_promotion = self.mcts.choose_move_and_update_graph(is_training=self.training)
        else:
            raise ValueError("No nodes to choose from.")
        return a2c_move, a2c_promotion

    def decide_move(self, board_state):
        board_tensor, _, _ = board_state
        if self.use_mcts:
            # Choose a move
            if self.whites_move:
                a2c_move, a2c_promotion = self._decide_move_for_player(board_tensor, self.running_white_states, self.running_white_prob)
                # Next to do 24/07/2026:
                # Speed up generating graph even more, think of ways
                # Fix any errors
                # Handling of draws by threefold repetition and 50 move rule will need to be decided with a meta-layer, since they are ignored during exploration.
                # Test games with white and black vs. Stockfish
                # Randomly flip half of the states when training
                # Save visualisations of the graph structure, including the probabilities and main constants per move. Test with mate in 1 and mate in 2 situations, and situations that look favourable for black.
                # For ^, Treelib/Graphvis? https://stackoverflow.com/questions/7670280/tree-plotting-in-python
                # Fix the new bug around pawn promotions and expanding valid probs (may have been fixed?)
                # Train the game value on pre-existing datasets
            else:
                a2c_move, a2c_promotion = self._decide_move_for_player(board_tensor, self.running_black_states, self.running_black_prob)
            self.boards.update_batch_size(1)  # To be safe.
        else:
            a2c_move, a2c_promotion = self.model.get_simple_max_moves(board_tensor)
        return (a2c_move, a2c_promotion)

    def prepare_for_training(self):
        self.training = True
        self.model.set_train_mode()

    def prepare_for_evaluation(self):
        self.training = False
        self.model.set_test_mode()

    def load_all_models(self):
        self.model.load_models('train')

    def analyse_loaded_data_and_models(self):
        # A sandbox area to look more closely at previous data or models.
        pass

    def log_opponent_move(self, move, promotion, board_state):
        # A generic component of log_stockfish_move
        # Reset the graph between moves when running against an opponent that is not myself.
        self.mcts.top_node = None
        self.mcts.nodes_marked_for_generation.clear()
        self.mcts.nodes_marked_for_rollout.clear()
        self.mcts.reset_graph()

    def log_stockfish_move(self, move, board_state, starting_colour_me):
        jle_move, jle_promotion = convert_UCI_to_jle_notation(move, starting_colour_me)
        self.log_opponent_move(jle_move, jle_promotion, board_state)
        return self.enact_move((jle_move, jle_promotion), board_state)

    def self_play_session(self):
        """
        Plays a game against itself (a full episode).
        """
        self.boards.update_batch_size(1)
        batched_board = self.boards.to_tensor().cuda()
        dud_move_count = self.boards.get_starting_move_count_list()
        colour_list = torch.ones((1)).to(torch.bool).cuda()

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

        game_over_message = get_game_over_message(game_over_tensor[0], not self.whites_move)
        if game_over_message is None:
            game_over_message = "Max game length reached. Terminating game"
        print(f"{game_over_message}, game length: {game_length}")
        episode_length = 0
        major_outcomes = game_over_tensor[0, 0:3]

        return major_outcomes, colour_list, states, moves, promotions

    def training_session(self, start_epoch, num_logged_games, memory):
        """
        Learn on the data that we have gathered so far.
        """
        self.model.set_train_mode()
        current_epoch = self.model.train_on_data(start_epoch, memory)
        self.update_training_params(num_logged_games)
        self.model.set_test_mode()
        return current_epoch

    def train_on_loaded_data(self, memory):
        self.model.set_train_mode()
        self.model.train_on_data(0, memory)


def train_single_threaded_a2c(artifacts_dir, model, memory, board_setup):
    num_logged_games = 0
    total_games = 0
    current_training_epoch = 0

    boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, board_setup)
    assert boards.get_batch_size() == 1
    batched_board = boards.to_tensor().cuda()
    starting_position = batched_board[0].clone()
    our_ai_agent = A2CMoveAgent(boards, model, starting_position, {}, True)
    our_ai_agent.prepare_for_training()

    while num_logged_games < TOTAL_DESIRED_LOGGED_GAMES_A2C:
        major_outcomes, colour_list, states, moves, promotions = our_ai_agent.self_play_session()
        save_full_game_artifacts(artifacts_dir, total_games + 1, states, moves, promotions)
        if torch.any(major_outcomes):
            true_game_value = get_game_value_for_white(major_outcomes, colour_list[0]).detach()
            memory.save_game_to_memory(
                true_game_value,
                our_ai_agent.running_white_states,
                our_ai_agent.running_black_states,
                our_ai_agent.running_white_prob,
                our_ai_agent.running_black_prob,
            )
            num_logged_games += 1
            print(f"{num_logged_games} logged games")
            if num_logged_games % A2C_TRAIN_CADENCE == 0:
                current_training_epoch = model.training_session(current_training_epoch, num_logged_games, memory)
        our_ai_agent.end_episode()
        total_games += 1
        print(total_games)


def multi_threaded_a2c_process(proc_num, model_state, artifacts_dir, board_setup, total_games):
    print(f"Starting process {proc_num}")
    boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, board_setup)
    assert boards.get_batch_size() == 1
    batched_board = boards.to_tensor().cuda()
    starting_position = batched_board[0].clone()
    model = A2CChessNetwork(model_state=model_state)
    our_ai_agent = A2CMoveAgent(boards, model, starting_position, {}, True)
    our_ai_agent.prepare_for_training()
    major_outcomes, colour_list, states, moves, promotions = our_ai_agent.self_play_session()
    # total_games should refer to the finished game count before any processes are started.
    save_full_game_artifacts(artifacts_dir, total_games + proc_num, states, moves, promotions)
    bs = our_ai_agent.running_black_states.copy()
    ws = our_ai_agent.running_white_states.copy()
    bp = our_ai_agent.running_black_prob.copy()
    wp = our_ai_agent.running_white_prob.copy()
    our_ai_agent.end_episode()
    return proc_num, major_outcomes, colour_list, ws, bs, wp, bp


def train_multi_threaded_a2c(artifacts_dir, model, memory, board_setup):
    num_logged_games = 0
    total_games = 0
    current_training_epoch = 0
    num_processes = 4  # 4 for full, 12 for simplified
    train_inc = 1
    agent_pool = mp.Pool(processes=num_processes)
    current_model_state = None

    while num_logged_games < TOTAL_DESIRED_LOGGED_GAMES_A2C:
        process_inputs = [(i, current_model_state, artifacts_dir, board_setup, total_games) for i in range(num_processes)]
        process_values = agent_pool.starmap(multi_threaded_a2c_process, process_inputs)
        for (proc_num, major_outcomes, col_list, ws, bs, wp, bp) in process_values:
            if torch.any(major_outcomes):
                true_game_value = get_game_value_for_white(major_outcomes, col_list[0]).detach()
                memory.save_game_to_memory(true_game_value, ws, bs, wp, bp)
                num_logged_games += 1
                print(f"{num_logged_games} logged games")
        torch.cuda.empty_cache()  # Clear up some VRAM, ideally
        if num_logged_games >= A2C_TRAIN_CADENCE * train_inc:
            current_training_epoch = model.training_session(current_training_epoch, num_logged_games, memory)
            train_inc += 1
        else:
            model.optimiser.zero_grad()
        total_games += num_processes
        print(f"Total games: {total_games}")


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
