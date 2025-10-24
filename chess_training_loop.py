# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:39:05 2023

@author: jledragon

Run this file with environment "jle".
"""

import torch
import chess_cpp
import time
from stockfish import Stockfish
from random import random
from abc import ABC, abstractmethod
import platform
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
    compile_if_supported
    #get_moves_for_player_with_reuse
)
from neural_networks import DQNChessNetwork

BATCH_SIZE = 256

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
    def _get_my_rewards(self, game_over_tensor, opponent_move_layer):
        pass
    
    @abstractmethod
    def _update_opponent_rewards(self, game_over_tensor, opponent_move_layer):
        pass
    
    def apply_all_rewards(self, game_over_tensor, game_state_bundle):
        rewards = self._get_my_rewards(game_over_tensor, game_state_bundle)
        self._update_opponent_rewards(game_over_tensor, game_state_bundle)
        return rewards
    
    @abstractmethod
    def store_training_artifacts(self, store_bundle, move_num):
        pass
    
    @abstractmethod
    def prepare_for_training(self):
        pass
    
    @abstractmethod
    def prepare_for_evaluation(self):
        pass
    
    @abstractmethod
    def train_step(self, epoch):
        pass
    
    @abstractmethod
    def save_all_models(self):
        pass
    
    @abstractmethod
    def load_all_models(self):
        pass

    def log_stockfish_move(self, move, board_state, starting_colour_me):
        """
        Record to the JLE game state what the Stockfish move was.
        """
        jle_move, jle_promotion = convert_UCI_to_jle_notation(move, starting_colour_me)
        return self.enact_move((jle_move, jle_promotion), board_state)
    
    def store_current_artifacts(self, current_state, action, rewards, terminals):
        self.previous_state = self.current_opponent_state
        self.previous_action = self.opponent_action
        self.previous_rewards = self.opponent_rewards
        self.previous_terminals = self.opponent_terminals
        self.current_opponent_state = current_state
        self.opponent_action = action
        self.opponent_rewards = rewards
        self.opponent_terminals = terminals
    
    @compile_if_supported
    def enact_move(self, move, board_state):
        randomly_selected_move, random_promotion = move
        board_tensor, colour_list, dud_move_count = board_state
        # Do the move
        chess_cpp.enact_moves(board_tensor, randomly_selected_move, random_promotion, dud_move_count)
        # Flip the board to opponent view
        flipped_board, inv_colour_list = flip_board(board_tensor, colour_list)
        # Log this position (white's view) for threefold repetition check
        white_view = get_white_view(board_tensor, flipped_board, colour_list)
        repetition_status = get_repetition_status(self.boards, white_view)
        colour_list = inv_colour_list
        # Get opponent's move to see if the game is over by the move we just did
        opponent_move_layer = chess_cpp.get_moves_for_player(flipped_board)
        # Check whether the game is over under any condition
        game_over_tensor = is_game_over(flipped_board, opponent_move_layer, repetition_status, dud_move_count)
        game_over = torch.any(game_over_tensor, dim=1)
        # Reset threefold repetitions list, move counts and colour wherever the game is over
        self.boards.reset_repetitions(game_over)
        reset_move_counts(dud_move_count, game_over)
        reset_colour_list(colour_list, game_over)
        # Flip the boards, reset wherever the game is over
        board_tensor = flipped_board
        board_tensor = possibly_reset_game(board_tensor, game_over, self.starting_position)
        
        return (randomly_selected_move, random_promotion), (dud_move_count, board_tensor, colour_list, opponent_move_layer, game_over_tensor)


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
            print(get_human_readable_board(board_tensor[0]))
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
    
    @compile_if_supported
    def _firepower_score_fn(self, board_state):
        if "firepower" in self.enabled_optional_rewards:
            my_firepower = get_firepower_score(board_state, False)
            opponent_firepower = get_firepower_score(board_state, True)
            scaled_difference_firepower = (my_firepower - opponent_firepower) * 0.1
            score = 2 * torch.sigmoid(scaled_difference_firepower) - 1  # Range between -1 and 1.
            return score
        else:
            return torch.zeros(board_state.shape[0]).to(board_state.device)
    
    @compile_if_supported
    def _move_reward_fn(self, opponent_move_layer):
        if "num_moves" in self.enabled_optional_rewards:
            sum_moves = torch.sum(opponent_move_layer, (1, 2))
            move_reward = self.move_reward_factor / (torch.sqrt(sum_moves + 1))
            # Do not count checkmate rewards twice in parallel with win/lose rewards.
            move_reward = torch.where(sum_moves == 0, 0, move_reward)
            return move_reward
        else:
            return torch.zeros(opponent_move_layer.shape[0]).to(opponent_move_layer.device)

    @compile_if_supported
    def _firepower_per_num_moves_fn(self, game_state_bundle):
        board_state, opponent_move_layer = game_state_bundle
        if "firepower_per_num_moves" in self.enabled_optional_rewards:
            opponent_firepower = get_firepower_score(board_state, True) + 4  # 4 is the King's firepower.
            sum_moves = torch.sum(opponent_move_layer, (1, 2))
            fp_np = torch.sqrt(opponent_firepower / (sum_moves + 1))
            return fp_np
        else:
            return torch.zeros(board_state.shape[0]).to(board_state.device)
    
    @compile_if_supported
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
    
    @compile_if_supported
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
                group['lr'] = 1e-5

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


class RandomMoveAgent(JLEAIMoveAgent):
    """
    An agent that picks a move randomly.
    """

    def __init__(self, boards, starting_position):
        self.boards = boards
        self.starting_position = starting_position
    
    @compile_if_supported
    def decide_and_enact_move(self, board_state):
        return super().decide_and_enact_move(board_state)
    
    def _get_my_rewards(self, game_over_tensor, game_state_bundle):
        rewards = torch.where(game_over_tensor[:,0], 100, -1)
        return rewards
    
    def _update_opponent_rewards(self, game_over_tensor, game_state_bundle):
        pass
    
    def store_training_artifacts(self, random_store_bundle, move_num):
        pass
    
    def prepare_for_training(self):
        pass
    
    def prepare_for_evaluation(self):
        pass
    
    def train_step(self, epoch):
        pass

    @compile_if_supported
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

    stockfish_agent.set_elo(0)
    for i in range(0, 20):
        stockfish_agent.start_new_game()
        dud_move_count = boards.get_starting_move_count_list()
        colour_list = torch.ones((1)).to(torch.bool).cuda()
        single_board = boards.to_tensor().cuda()
        winner = ''
        if random() > 0.5:
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


@compile_if_supported
def evaluate_against_random(boards, random_agent, our_ai_agent):
    """
    Play an AI against random, to test whether a new training algorithm is working.
    """
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
    

def ai_training_loop(boards, our_ai_agent, start_epoch):
    """
    Play an AI against itself. Generally for training.
    """
    batched_board = boards.to_tensor().cuda()
    dud_move_count = boards.get_starting_move_count_list()
    colour_list = torch.ones((BATCH_SIZE)).to(torch.bool).cuda()
    total_games = 0
    
    #now = time.time()
    end_epoch = start_epoch + 100
    for move_num in range(start_epoch, end_epoch):
        current_state = batched_board.clone()
        (move, promotion), (dud_move_count, batched_board, colour_list, opponent_move_layer, game_over_tensor) = \
            our_ai_agent.decide_and_enact_move((batched_board, colour_list, dud_move_count))
        rewards = our_ai_agent.apply_all_rewards(game_over_tensor, (batched_board, opponent_move_layer))
        game_over = torch.any(game_over_tensor, dim=1)
        # Store state, action, rewards, terminals and next state to the buffer.
        our_ai_agent.store_training_artifacts((current_state, move, promotion, rewards, game_over), opponent_move_layer)
        if move_num >= 100:
            our_ai_agent.train_step(move_num)
        games_done = torch.sum(game_over.to(torch.int8))
        total_games += games_done
        #print(move_num)
    #elapsed = time.time() - now
    #print(elapsed)
    return end_epoch
    

if __name__ == '__main__':
    # Starting condition
    torch._dynamo.config.cache_size_limit = 256
    boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, 0)
    batched_board = boards.to_tensor().cuda()
    starting_position = batched_board[0].clone()
    stockfish_agent = StockfishMoveAgent(boards, starting_position)
    our_ai_agent = DQNMoveAgent(boards, starting_position, {})  # {"firepower", "firepower_per_num_moves"}
    #our_ai_agent.load_all_models()
    random_agent = RandomMoveAgent(boards, starting_position)
    our_ai_agent.prepare_for_training()
    start_epoch = 0
    start_epoch = ai_training_loop(boards, our_ai_agent, start_epoch)
    for i in range(0, 20):
        if i == 1:
            now = time.time()
        our_ai_agent.prepare_for_training()
        start_epoch = ai_training_loop(boards, our_ai_agent, start_epoch)
        print(our_ai_agent.q_network.eps, start_epoch)
        our_ai_agent.prepare_for_evaluation()
        evaluate_against_random(boards, random_agent, our_ai_agent)
        if i == 1:
            print(time.time() - now)
    our_ai_agent.prepare_for_evaluation()
    evaluate_against_stockfish(boards, stockfish_agent, our_ai_agent)
    our_ai_agent.save_all_models()
