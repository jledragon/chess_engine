# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:08:03 2022

@author: jledragon

Run "python setup.py install --user, pip install -e ." to build all C++ utils.
std::cout << "str" works as a python print statement when run in C++.
"""

import torch
from torch.nn.functional import one_hot
import chess_cpp
from bidict import bidict
from string import Template


_cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
_reverse_cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
_promotion_indices = bidict({'n': 0, 'b': 1, 'r': 2, 'q': 3})

def compile_if_supported(func):
    try:
        # This pattern is fast when compile is valid, i.e. when we would want it to be fast.
        @torch.compile
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    except RuntimeError:
        # More general - don't tie it to OS/architecture, simply whether compile is supported or not.
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    # All other errors will raise an Exception as usual.
    return wrapper

def get_game_over_message(game_over_for_batch, colour_list):
    # For use in visualisation only.
    if not torch.any(game_over_for_batch):
        return None
    else:
        player = "white" if colour_list else "black"
        if game_over_for_batch[0]:
            return f"Game over - {player} wins!"
        elif game_over_for_batch[1]:
            return "It's a draw by stalemate!"
        elif game_over_for_batch[2]:
            return "Insufficient material left of the board. It's a draw!"
        elif game_over_for_batch[3]:
            return "It's a draw by threefold repetition!"
        elif game_over_for_batch[4]:
            return "No progress made in a while. It's a draw!"

def get_all_binary_moves(batch_size):
    all_moves = torch.arange(0, 4096).unsqueeze(0).cuda()
    all_moves = all_moves.repeat(batch_size, 1)
    mask = 2**torch.arange(11, -1, -1).to(all_moves.device, all_moves.dtype)
    return all_moves.unsqueeze(-1).bitwise_and(mask).ne(0).float()

@compile_if_supported
def flip_board(batched_board, colour_list):
    # Flip the colours anywhere we have a piece
    # True is white, False is black.
    flipped = torch.where(
        (batched_board[:, 0, :, :] == 1) |
        (batched_board[:, 1, :, :] == 1) |
        (batched_board[:, 2, :, :] == 1) |
        (batched_board[:, 3, :, :] == 1) |
        (batched_board[:, 4, :, :] == 1),
        1 - batched_board[:, 5, :, :],
        batched_board[:, 5, :, :]
    )
    return_board = torch.clone(batched_board)
    return_board[:, 5, :, :] = flipped
    # Turn the board around vertically
    return_board = torch.flip(return_board, (2, 3))
    return return_board, torch.logical_not(colour_list)

@compile_if_supported
def has_insufficient_mating_material(batched_board):
    # All sufficient rules are irregardless of colour.
    sufficient_1 = torch.sum(
        # Anywhere we have a pawn or rook (or queen).
        torch.where(
            (batched_board[:, 0, :, :] == 1) |
            (batched_board[:, 3, :, :] == 1),
            1, 0
        ), (1, 2)
    ) > 0
    flat_bishop_view = batched_board.reshape(batched_board.shape[0], batched_board.shape[1], batched_board.shape[2] * batched_board.shape[3])[:, 2, :]
    light_bishop = flat_bishop_view[:, ::2]
    dark_bishop = flat_bishop_view[:, 1::2]
    # At least one light squared bishop with at least one dark squared bishop is sufficient.
    sufficient_2 = (torch.sum(light_bishop, 1) > 0) & (torch.sum(dark_bishop, 1) > 0)
    knight_view = batched_board[:, 1, :, :]
    # More than one knight is sufficient.
    sufficient_3 = torch.sum(knight_view, (1, 2)) > 1
    sufficient_4 = (torch.sum(knight_view, (1, 2)) > 0) & (torch.sum(flat_bishop_view, 1) > 0)
    insufficient = torch.logical_not(
        (sufficient_1 | sufficient_2 | sufficient_3 | sufficient_4)
    )
    return insufficient

@compile_if_supported
def get_white_view(batched_board, flipped_board, colour_list):
    exp_col_list = colour_list.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    white_view = torch.where(
        exp_col_list, batched_board, flipped_board
    )
    return white_view

@compile_if_supported
def get_repetition_status(boards, batched_board):
    p64 = torch.ones((batched_board.shape[0], 1, batched_board.shape[2], batched_board.shape[3])).cuda()
    p128 = torch.zeros((batched_board.shape[0], 1, batched_board.shape[2], batched_board.shape[3])).cuda()
    char_pos = torch.cat((batched_board[:, 0:6, :, :], p64, p128), dim=1)
    sq_num = torch.arange(0, 8).cuda()
    power_mask = torch.pow(2, sq_num).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    compact_pos = torch.sum(char_pos * power_mask, 1).to(torch.int8).cpu()
    repetition_status = boards.check_threefold_repetition(compact_pos)
    return repetition_status

@compile_if_supported
def get_single_board_encoding(batched_board):
    assert batched_board.shape[0] == 1, "Only boards with batch size 1 are supported."
    p64 = torch.ones((batched_board.shape[0], 1, batched_board.shape[2], batched_board.shape[3])).cuda()
    p128 = torch.zeros((batched_board.shape[0], 1, batched_board.shape[2], batched_board.shape[3])).cuda()
    char_pos = torch.cat((batched_board[:, 0:6, :, :], p64, p128), dim=1)
    sq_num = torch.arange(0, 8).cuda()
    power_mask = torch.pow(2, sq_num).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    compact_pos = torch.sum(char_pos * power_mask, 1).to(torch.int8).cpu()
    board_encoding = chess_cpp.get_board_encoding(compact_pos, 0)  # Just check the first element
    return board_encoding

@compile_if_supported
def is_game_over(batched_board, move_layer, repetition_status, dud_move_count):
    num_moves = torch.sum(move_layer, (1, 2))
    game_over = (num_moves == 0)
    is_stalemate = chess_cpp.is_stalemate(batched_board, game_over)
    is_checkmate = game_over.unsqueeze(1) & torch.logical_not(is_stalemate)
    has_insufficient_material = has_insufficient_mating_material(batched_board).unsqueeze(1)
    getting_nowhere = dud_move_count >= 50
    game_state_per_board = torch.cat((is_checkmate, is_stalemate, has_insufficient_material, repetition_status, getting_nowhere), dim=1)
    return game_state_per_board

@compile_if_supported
def get_moves_for_player_with_reuse(batched_board, game_over, opponent_move_layer):
    move_layer = chess_cpp.get_moves_for_player_with_skip(batched_board, torch.logical_not(game_over))
    move_layer_reuse = torch.where(
        game_over.reshape((game_over.shape[0], 1, 1)),
        move_layer,
        opponent_move_layer
    )
    return move_layer_reuse

@compile_if_supported
def get_random_move(batched_board, move_layer):
    num_moves = torch.sum(move_layer, (1, 2))
    random_move = (torch.rand(size=num_moves.shape).cuda() * num_moves).to(torch.int8)
    flat_move_layer = move_layer.reshape((batched_board.shape[0], -1))
    random_select = chess_cpp.get_random_valid_move_per_game(flat_move_layer, random_move)
    random_promotion = (torch.rand(size=num_moves.shape).cuda() * 4).to(torch.int64)
    one_hot_promotion = one_hot(random_promotion, num_classes=4).to(torch.int8)
    return random_select, one_hot_promotion

@compile_if_supported
def flip_episode(states, actions, next_states):
    indices_to_flip = torch.rand(size=(states.shape[0],)).cuda() > 0.5
    states_to_flip = states[indices_to_flip]
    flipped_states = torch.flip(states_to_flip, (3,))
    action_move = actions[0].clone()
    actions_to_flip = action_move[indices_to_flip]
    actions_to_flip[:,1] = 7 - actions_to_flip[:,1]
    actions_to_flip[:,3] = 7 - actions_to_flip[:,3]
    flipped_actions = actions_to_flip
    next_states_to_flip = next_states[indices_to_flip]
    flipped_next_states = torch.flip(next_states_to_flip, (3,))
    states[indices_to_flip] = flipped_states
    action_move[indices_to_flip] = flipped_actions
    next_states[indices_to_flip] = flipped_next_states
    return states, (action_move, actions[1]), next_states

@compile_if_supported
def get_firepower_score(batched_board, for_opponent):
    # Careful - this will deal with a flipped view.
    player_num = 0 if for_opponent else 1
    firepower_board = torch.where(
        (batched_board[:,0,:,:] == 1) &
        (batched_board[:,5,:,:] == player_num),
        1, 0 # Pawn
    )
    firepower_board += torch.where(
        (batched_board[:,1,:,:] == 1) &
        (batched_board[:,5,:,:] == player_num),
        3, 0 # Knight
    )
    firepower_board += torch.where(
        (batched_board[:,2,:,:] == 1) &
        (batched_board[:,5,:,:] == player_num),
        3, 0 # Bishop
    )
    firepower_board += torch.where(
        (batched_board[:,3,:,:] == 1) &
        (batched_board[:,5,:,:] == player_num),
        5, 0 # Rook
    )
    firepower_board += torch.where(
        (batched_board[:,2,:,:] == 1) &
        (batched_board[:,3,:,:] == 1) &
        (batched_board[:,5,:,:] == player_num),
        1, 0 # Queen (this is in addition to what already exists, mind)
    )
    batched_firepower = torch.sum(firepower_board, (1,2))
    return batched_firepower

@compile_if_supported
def reset_move_counts(move_count, is_game_over):
    move_count[is_game_over, :] = 0
    return move_count

@compile_if_supported
def reset_colour_list(colour_list, is_game_over):
    colour_list[is_game_over] = True
    return colour_list

@compile_if_supported
def possibly_reset_game(batched_board, is_game_over, starting_position):
    batched_board[is_game_over] = starting_position
    return batched_board

@compile_if_supported
def expand_all_moves(single_board, move_layer_for_board):
    flat_moves = move_layer_for_board.reshape((move_layer_for_board.shape[0] * move_layer_for_board.shape[1]))
    legal_moves = torch.where(flat_moves == 1)[0].unsqueeze(1)
    ft1 = legal_moves // 64
    ft2 = legal_moves % 64
    f1 = ft1 // 8
    t1 = ft1 % 8
    f2 = ft2 // 8
    t2 = ft2 % 8
    all_possible_moves = torch.cat((f1, t1, f2, t2), dim=1).to(torch.int8)
    copied_boards = single_board.unsqueeze(0).repeat(all_possible_moves.shape[0], 1, 1, 1)
    is_promotion = chess_cpp.get_pawn_promote_move_mask(copied_boards, all_possible_moves)
    default_promotion = torch.Tensor([[0, 0, 0, 1]]).repeat(all_possible_moves.shape[0], 1).to(torch.int8).cuda()
    expanded_boards = chess_cpp.expand_boards(copied_boards, is_promotion)
    expanded_promotions = chess_cpp.expand_promotions(default_promotion, is_promotion)
    return expanded_boards, expanded_promotions

def convert_jle_to_UCI_notation(move, promotion, single_board, invert):
    assert len(move) == 1  # Only do this in Python for a single batch.
    # Get the pawn promotion part, if applicable
    enc = single_board[0, :, move[0][0], move[0][1]]
    piece_enc_sum = torch.sum(enc[0:5])
    promo_letter = ""
    if enc[0] == 1 and piece_enc_sum == 1 and move[0][2] == 7:
        promo_index = torch.nonzero(promotion)
        assert promo_index.shape[1] == 1
        promo_letter = _promotion_indices.inverse(promo_index[0][0])
    if invert:
        move = 7 - move
    return f"{_cols[move[0][1]]}{move[0][0]+1}{_cols[move[0][3]]}{move[0][2]+1}{promo_letter}"

def convert_UCI_to_jle_notation(move, invert):
    jle_move = torch.Tensor([[int(move[1])-1, _reverse_cols[move[0]], int(move[3])-1, _reverse_cols[move[2]]]]).to(torch.int8).cuda()
    if invert:
        jle_move = 7 - jle_move
    jle_promotion = torch.Tensor([0, 0, 0, 0]).to(torch.int8).cuda()
    if len(move) > 4:
        jle_promotion[_promotion_indices[move[4]]] = 1
    return jle_move, jle_promotion

def get_human_readable_board(single_board_element):
    board_str = Template("""
    +---+---+---+---+---+---+---+---+
    | $a8 | $b8 | $c8 | $d8 | $e8 | $f8 | $g8 | $h8 | 8
    +---+---+---+---+---+---+---+---+
    | $a7 | $b7 | $c7 | $d7 | $e7 | $f7 | $g7 | $h7 | 7
    +---+---+---+---+---+---+---+---+
    | $a6 | $b6 | $c6 | $d6 | $e6 | $f6 | $g6 | $h6 | 6
    +---+---+---+---+---+---+---+---+
    | $a5 | $b5 | $c5 | $d5 | $e5 | $f5 | $g5 | $h5 | 5
    +---+---+---+---+---+---+---+---+
    | $a4 | $b4 | $c4 | $d4 | $e4 | $f4 | $g4 | $h4 | 4
    +---+---+---+---+---+---+---+---+
    | $a3 | $b3 | $c3 | $d3 | $e3 | $f3 | $g3 | $h3 | 3
    +---+---+---+---+---+---+---+---+
    | $a2 | $b2 | $c2 | $d2 | $e2 | $f2 | $g2 | $h2 | 2
    +---+---+---+---+---+---+---+---+
    | $a1 | $b1 | $c1 | $d1 | $e1 | $f1 | $g1 | $h1 | 1
    +---+---+---+---+---+---+---+---+
      a   b   c   d   e   f   g   h
    """)
    subst_dict = {}
    for col in range(0, 8):
        for row in range(0, 8):
            letter = " "
            if single_board_element[5][col][row] == 1:
                if single_board_element[0][col][row] == 1:
                    letter = 'p'
                elif single_board_element[1][col][row] == 1:
                    letter = 'n'
                elif single_board_element[2][col][row] == 1 and single_board_element[3][col][row] == 1:
                    letter = 'q'
                elif single_board_element[2][col][row] == 1:
                    letter = 'b'
                elif single_board_element[3][col][row] == 1:
                    letter = 'r'
                elif single_board_element[4][col][row] == 1:
                    letter = 'k'
            else:
                if single_board_element[0][col][row] == 1:
                    letter = 'P'
                elif single_board_element[1][col][row] == 1:
                    letter = 'N'
                elif single_board_element[2][col][row] == 1 and single_board_element[3][col][row] == 1:
                    letter = 'Q'
                elif single_board_element[2][col][row] == 1:
                    letter = 'B'
                elif single_board_element[3][col][row] == 1:
                    letter = 'R'
                elif single_board_element[4][col][row] == 1:
                    letter = 'K'
            subst_dict[f"{_cols[row]}{col + 1}"] = letter
    return board_str.substitute(subst_dict)
