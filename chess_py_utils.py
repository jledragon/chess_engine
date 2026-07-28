# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:08:03 2022

@author: jledragon

Run "python setup.py install --user, pip install -e ." to build all C++ utils.
If "pip install -e ." does not work, try "pip install --no-build-isolation -e ."
Close Anaconda if permission is denied when running the second command.
std::cout << "str" works as a python print statement when run in C++.
"""

import torch
from torch.nn.functional import one_hot
import chess_cpp
from bidict import bidict
from string import Template
import os


_cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
_reverse_cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
_promotion_indices = bidict({'n': 0, 'b': 1, 'r': 2, 'q': 3})


def conditional_compile(func):
    if os.environ.get('try_compile', 'False').lower() == 'true':
        @torch.compile
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    else:
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
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

def get_game_value_for_white(major_outcomes_for_batch, colour_list):
    # TODO - make one for black also
    if major_outcomes_for_batch[0] and colour_list:
        return torch.Tensor([1]).to(torch.float32).cuda()
    elif major_outcomes_for_batch[0] and not colour_list:
        return torch.Tensor([-1]).to(torch.float32).cuda()
    elif torch.any(major_outcomes_for_batch[1:3]):
        return torch.Tensor([0]).to(torch.float32).cuda()

def get_all_binary_moves(batch_size):
    all_moves = torch.arange(0, 4096).unsqueeze(0).cuda()
    all_moves = all_moves.repeat(batch_size, 1)
    mask = 2**torch.arange(11, -1, -1).to(all_moves.device, all_moves.dtype)
    return all_moves.unsqueeze(-1).bitwise_and(mask).ne(0).float()

@conditional_compile
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

@conditional_compile
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

@conditional_compile
def get_white_view(batched_board, flipped_board, colour_list):
    exp_col_list = colour_list.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    white_view = torch.where(
        exp_col_list, batched_board, flipped_board
    )
    return white_view

@conditional_compile
def get_repetition_status(boards, batched_board):
    p64 = torch.ones((batched_board.shape[0], 1, batched_board.shape[2], batched_board.shape[3])).cuda()
    p128 = torch.zeros((batched_board.shape[0], 1, batched_board.shape[2], batched_board.shape[3])).cuda()
    char_pos = torch.cat((batched_board[:, 0:6, :, :], p64, p128), dim=1)
    sq_num = torch.arange(0, 8).cuda()
    power_mask = torch.pow(2, sq_num).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    compact_pos = torch.sum(char_pos * power_mask, 1).to(torch.int8).cpu()
    repetition_status = boards.check_threefold_repetition(compact_pos)
    return repetition_status

@conditional_compile
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

@conditional_compile
def is_game_over(batched_board, move_layer, repetition_status, dud_move_count):
    num_moves = torch.sum(move_layer, (1, 2))
    game_over = (num_moves == 0)
    is_stalemate = chess_cpp.is_stalemate(batched_board, game_over)
    is_checkmate = game_over.unsqueeze(1) & torch.logical_not(is_stalemate)
    has_insufficient_material = has_insufficient_mating_material(batched_board).unsqueeze(1)
    getting_nowhere = dud_move_count >= 50
    game_state_per_board = torch.cat((is_checkmate, is_stalemate, has_insufficient_material, repetition_status, getting_nowhere), dim=1)
    return game_state_per_board

@conditional_compile
def get_moves_for_player_with_reuse(batched_board, game_over, opponent_move_layer):
    move_layer = chess_cpp.get_moves_for_player_with_skip(batched_board, torch.logical_not(game_over))
    move_layer_reuse = torch.where(
        game_over.reshape((game_over.shape[0], 1, 1)),
        move_layer,
        opponent_move_layer
    )
    return move_layer_reuse

@conditional_compile
def get_random_move(batched_board, move_layer):
    num_moves = torch.sum(move_layer, (1, 2))
    random_move = (torch.rand(size=num_moves.shape).cuda() * num_moves).to(torch.int8)
    flat_move_layer = move_layer.reshape((batched_board.shape[0], -1))
    random_select = chess_cpp.get_random_valid_move_per_game(flat_move_layer, random_move)
    random_promotion = (torch.rand(size=num_moves.shape).cuda() * 4).to(torch.int64)
    one_hot_promotion = one_hot(random_promotion, num_classes=4).to(torch.int8)
    return random_select, one_hot_promotion

@conditional_compile
def flip_states(states):
    indices_to_flip = torch.rand(size=(states.shape[0],)).cuda() > 0.5
    states_to_flip = states[indices_to_flip]
    flipped_states = torch.flip(states_to_flip, (3,))
    states[indices_to_flip] = flipped_states
    return states

@conditional_compile
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

@conditional_compile
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

@conditional_compile
def reset_move_counts(move_count, is_game_over):
    move_count[is_game_over, :] = 0
    return move_count

@conditional_compile
def reset_colour_list(colour_list, is_game_over):
    colour_list[is_game_over] = True
    return colour_list

@conditional_compile
def possibly_reset_game(batched_board, is_game_over, starting_position):
    batched_board[is_game_over] = starting_position
    return batched_board

@conditional_compile
def expand_valid_probs(single_board, valid_probs, is_promotion, val_prom):
    if valid_probs.shape[0] == 0 or torch.sum(is_promotion) == 0:
        return valid_probs  # Shortcut
    # This method does several things:
    # 1 - everywhere where the move is a promotion, expand the probability vector 4x
    # 2 - apply the promotion softmax to these four entries - that all probabilities add up to 1 is assumed.
    #     These four entries should now add up to the unexpanded value.
    # 3 - return the new, expanded Tensor
    vp_expanded = valid_probs.unsqueeze(1)
    # Expand the probabilities, so each one is repeated 4x in a new dimension
    vp_expanded = vp_expanded.expand(valid_probs.shape[0], 4)
    prom_expanded = is_promotion.unsqueeze(1)
    # Zero any repeated probability that's not part of a promotion. We will want to drop it later.
    ignore_prob_mask = torch.cat(
        (
            torch.zeros(prom_expanded.shape[0], 1).to(torch.bool).to(val_prom.device),
            torch.logical_not(prom_expanded.expand(prom_expanded.shape[0], 3)
        )
    ), dim=1)
    # Multiply the promotion vector by the expanded probability. E.g. if we had 0.3, which was expanded
    # to [0.3, 0.3, 0.3, 0.3], along with softmax promotion for the move (0.1, 0.1, 0.7, 0.1), multiply
    # these together to get [0.03, 0.03, 0.21, 0.03]
    valid_probs_prom_applied = torch.where(
        prom_expanded,
        vp_expanded * val_prom,
        vp_expanded
    )
    # Zero anything we don't want
    # Add 1 to everything for safety. We are essentially setting -1 to anything we don't want, but
    # still want to use nonzero(). -1 will not be seen anywhere else, since it's a probability
    wanted_probs = torch.where(ignore_prob_mask, 0, valid_probs_prom_applied + 1).flatten()
    # Filter the list to finish.
    filtered_probs = wanted_probs[wanted_probs.nonzero()][:,0] - 1
    return filtered_probs

@conditional_compile
def expand_all_boards(all_boards, num_moves_per_batch_element):
    all_expanded_boards = []
    for board_i in range(all_boards.shape[0]):
        # This is the part that needs to work with single elements,
        # as `num_moves_per_batch_element` is different per index.
        all_expanded_boards.append(all_boards[board_i].unsqueeze(0).repeat(num_moves_per_batch_element[board_i], 1, 1, 1))
    all_expanded_boards = torch.cat(all_expanded_boards, axis=0)
    return all_expanded_boards

@conditional_compile
def get_new_batch_splits(is_promotion, num_moves_per_batch_element):
    total = 0
    splits = []
    for b_num in num_moves_per_batch_element:
        is_prom_slice = is_promotion[total:total+b_num]
        total += b_num
        splits.append(b_num + 3 * torch.unsqueeze(torch.sum(is_prom_slice), 0))
    return torch.cat(splits).to(torch.int32).cuda()

@conditional_compile
def get_new_prom_splits(num_moves_per_batch_element, valid_t2, out_prom):
    total = 0
    splits = []
    for i, b_num in enumerate(num_moves_per_batch_element):
        prom_slice = out_prom[i, valid_t2[total:total+b_num], :]
        splits.append(prom_slice.squeeze(1))
    return torch.cat(splits).cuda()

@conditional_compile
def expand_all_moves(all_boards, softmax_prom, legal_moves, valid_probs, num_moves_per_batch_element):
    assert all_boards.shape[1] == 8, "Remember to use the full board here."
    ft1 = legal_moves // 64
    ft2 = legal_moves % 64
    f1 = ft1 // 8
    t1 = ft1 % 8
    f2 = ft2 // 8
    t2 = ft2 % 8
    all_possible_moves = torch.cat((f1, t1, f2, t2), dim=1).to(torch.int8)
    copied_boards = expand_all_boards(all_boards, num_moves_per_batch_element)
    assert copied_boards.shape[0] <= 65_536, "Too many copied boards."  # Safety - this can just be made larger in CPP. It is 256 x 256.
    is_promotion = chess_cpp.get_pawn_promote_move_mask(copied_boards, all_possible_moves)
    if torch.sum(is_promotion) > 0:
        val_prom = get_new_prom_splits(num_moves_per_batch_element, t2, softmax_prom)
        expanded_boards = chess_cpp.expand_boards(copied_boards, is_promotion)
        default_promotion = torch.Tensor([[0, 0, 0, 1]]).repeat(is_promotion.shape[0], 1).to(torch.int8).cuda()
        expanded_promotions = chess_cpp.expand_promotions(default_promotion, is_promotion)
        expanded_moves = chess_cpp.expand_moves(all_possible_moves, is_promotion)
        new_splits = get_new_batch_splits(is_promotion, num_moves_per_batch_element)
        # Keep gradients here
        expanded_valid_probs = expand_valid_probs(all_boards, valid_probs, is_promotion, val_prom)
        return expanded_boards, expanded_moves, expanded_promotions, expanded_valid_probs, new_splits
    else:
        # No promotions, so this can just be anything.
        default_promotion = torch.Tensor([[0, 0, 0, 1]]).repeat(is_promotion.shape[0], 1).to(torch.int8).cuda()
        return copied_boards, all_possible_moves, default_promotion, valid_probs, num_moves_per_batch_element

def convert_jle_to_UCI_notation(move, promotion, single_board, invert):
    assert len(move) == 1  # Only do this in Python for a single batch.
    # Get the pawn promotion part, if applicable
    enc = single_board[0, :, move[0][0], move[0][1]]
    piece_enc_sum = torch.sum(enc[0:5])
    promo_letter = ""
    if enc[0] == 1 and piece_enc_sum == 1 and move[0][2] == 7:
        if len(promotion.shape) == 2 and promotion.shape[0] == 1:
            promotion = promotion[0]
        promo_index = torch.argwhere(promotion)
        assert promo_index.shape[1] == 1
        promo_letter = _promotion_indices.inverse[promo_index[0][0].item()]
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

def projection_fn(from_row, from_col, to_row, to_col):
    return from_row * 8**3 + from_col * 8**2 + to_row * 8 + to_col

def get_diagonal_projections(from_row, from_col, to_row):
    projections = []
    diff = from_row - to_row if from_row > to_row else to_row - from_row
    to_col_1 = from_col - diff
    to_col_2 = from_col + diff
    if to_col_1 >= 0:
        projections.append(projection_fn(from_row, from_col, to_row, to_col_1))
    if to_col_2 <= 7:
        projections.append(projection_fn(from_row, from_col, to_row, to_col_2))
    return projections

def get_legal_moves_projection():
    # Not all 4,096 moves are legal in chess. This method returns a positional tensor that can be used to populate a 4,096-sized
    # tensor with all actual, legal moves. This is a shortcut that should only be used with base/vanilla chess on an 8x8 board.
    positions = []
    for from_row in range(8):
        for from_col in range(8):
            for to_row in range(from_row - 2, from_row + 3):
                if to_row < 0 or to_row > 7:
                    continue  # Remove anything that goes out of bounds row-wise
                for to_col in range(from_col - 2, from_col + 3):
                    if to_col < 0 or to_col > 7:
                        continue  # Remove anything that goes out of bounds column-wise
                    if from_row == to_row and from_col == to_col:
                        continue  # No piece moves to its own current square.
                    positions.append(projection_fn(from_row, from_col, to_row, to_col))
            # Project all rows, columns and diagonals from here to the edge of the board. Skip the
            # first two squares in all cases, which will be accounted for already by the above code.
            for to_row in range(from_row - 2):
                positions.append(projection_fn(from_row, from_col, to_row, from_col))
                positions += get_diagonal_projections(from_row, from_col, to_row)
            for to_row in range(from_row + 3, 8):
                positions.append(projection_fn(from_row, from_col, to_row, from_col))
                positions += get_diagonal_projections(from_row, from_col, to_row)
            for to_col in range(from_col - 2):
                positions.append(projection_fn(from_row, from_col, from_row, to_col))
            for to_col in range(from_col + 3, 8):
                positions.append(projection_fn(from_row, from_col, from_row, to_col))
    positions = torch.Tensor(positions).to(torch.int32).unique().cuda()
    assert positions.shape[0] == 1_792  # If this is not correct, something has gone wrong in the logic.
    return positions

def save_full_game_artifacts(artifacts_dir, game_num, states, moves, promotions):
    game_file_name = artifacts_dir / f"game_{str(game_num)}"
    game_file = game_file_name.with_suffix(".txt")
    game_trail = "Start state, white to play:\n" + get_human_readable_board(states[0][0,:,:,:], True) + "\n"
    num_moves = len(moves)
    turn = True
    for move_num in range(num_moves):
        turn_str = f"{'black' if turn else 'white'} to play"
        readable_move = convert_jle_to_UCI_notation(moves[move_num], promotions[move_num], states[move_num], not turn)
        game_trail += f"Logged move: {readable_move}\n\n"
        current_state = states[move_num + 1]
        state_str = get_human_readable_board(current_state[0,:,:,:], not turn)
        game_trail += f"Current state, {turn_str}:\n{state_str}\n"
        turn = not turn
    game_file.write_text(game_trail, encoding="utf-8")

def get_mode_str(mode):
    match mode:
        case 0:
            return "full"
        case 1 | 2:
            return "simplified"
        case 3 | 4 | 5 | 6 | 7:
            return "puzzle"

def get_human_readable_board(single_board_element, as_white):
    if as_white:
        board_str = Template(
            """
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
            """
        )
    else:
        board_str = Template(
            """
              +---+---+---+---+---+---+---+---+
            1 | $a8 | $b8 | $c8 | $d8 | $e8 | $f8 | $g8 | $h8 |
              +---+---+---+---+---+---+---+---+
            2 | $a7 | $b7 | $c7 | $d7 | $e7 | $f7 | $g7 | $h7 |
              +---+---+---+---+---+---+---+---+
            3 | $a6 | $b6 | $c6 | $d6 | $e6 | $f6 | $g6 | $h6 |
              +---+---+---+---+---+---+---+---+
            4 | $a5 | $b5 | $c5 | $d5 | $e5 | $f5 | $g5 | $h5 |
              +---+---+---+---+---+---+---+---+
            5 | $a4 | $b4 | $c4 | $d4 | $e4 | $f4 | $g4 | $h4 |
              +---+---+---+---+---+---+---+---+
            6 | $a3 | $b3 | $c3 | $d3 | $e3 | $f3 | $g3 | $h3 |
              +---+---+---+---+---+---+---+---+
            7 | $a2 | $b2 | $c2 | $d2 | $e2 | $f2 | $g2 | $h2 |
              +---+---+---+---+---+---+---+---+
            8 | $a1 | $b1 | $c1 | $d1 | $e1 | $f1 | $g1 | $h1 |
              +---+---+---+---+---+---+---+---+
                h   g   f   e   d   c   b   a
            """
        )
    subst_dict = {}
    for col in range(0, 8):
        for row in range(0, 8):
            letter = " "
            if single_board_element[5][col][row] == 1:
                # White gets capitals, black gets lower case.
                if single_board_element[0][col][row] == 1:
                    letter = 'p' if as_white else 'P'
                elif single_board_element[1][col][row] == 1:
                    letter = 'n' if as_white else 'N'
                elif single_board_element[2][col][row] == 1 and single_board_element[3][col][row] == 1:
                    letter = 'q' if as_white else 'Q'
                elif single_board_element[2][col][row] == 1:
                    letter = 'b' if as_white else 'B'
                elif single_board_element[3][col][row] == 1:
                    letter = 'r' if as_white else 'R'
                elif single_board_element[4][col][row] == 1:
                    letter = 'k' if as_white else 'K'
            else:
                if single_board_element[0][col][row] == 1:
                    letter = 'P' if as_white else 'p'
                elif single_board_element[1][col][row] == 1:
                    letter = 'N' if as_white else 'n'
                elif single_board_element[2][col][row] == 1 and single_board_element[3][col][row] == 1:
                    letter = 'Q' if as_white else 'q'
                elif single_board_element[2][col][row] == 1:
                    letter = 'B' if as_white else 'b'
                elif single_board_element[3][col][row] == 1:
                    letter = 'R' if as_white else 'r'
                elif single_board_element[4][col][row] == 1:
                    letter = 'K' if as_white else 'k'
            subst_dict[f"{_cols[row]}{col + 1}"] = letter
    return board_str.substitute(subst_dict)
