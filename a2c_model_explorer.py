# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:45:42 2026

@author: jledragon

This script will open an interactive chessboard GUI with some starting position and with some loaded model.
It will be strictly human vs. human play and all legal 'to' positions will be marked with some shade of red,
depending on how much the model predicts that we should move there. These values will be normalised between
0-1, with 0 meaning 'add no red to the square' and 1 meaning the square should be solidly red. All legal
'from' pieces will be marked in blue in the same way, depending on their contribution. When selecting a piece,
all legal 'to' moves for just that piece will be highlighted. Nothing else is highlighted in red on blue in
this case. The predicted value for the position from the POV of black and of white will be shown underneath
the board. This will help us to debug what the models are learning.

"""

import torch
import chess_cpp
from constants import BATCH_SIZE
from chess_py_utils import (
    get_mode_str,
    flip_board,
    is_game_over,
    get_white_view,
    get_repetition_status,
    get_game_over_message,
    get_to_and_from_move_contributions
)
from neural_networks import A2CChessNetwork
from visualisation import BoardDrawer
import FreeSimpleGUI as sg


if __name__ == '__main__':
    # Insist that we have CUDA for now, otherwise things will be much slower.
    assert torch.cuda.is_available(), "CUDA is not enabled. Please fix this before running this script."
    torch._dynamo.config.cache_size_limit = 64
    mode = 1
    mode_str = get_mode_str(mode)
    boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, mode)
    model = A2CChessNetwork()
    model.load_models('eval')
    batched_board = boards.to_tensor()
    draw_tensor = batched_board[0, 0:6, :, :]
    colour_list = torch.Tensor([True]).to(torch.bool).cuda()
    cuda_board = batched_board.cuda()
    model_board = cuda_board[:, 0:6, :, :]
    move_layer = chess_cpp.get_moves_for_player(cuda_board)
    pred_act, pred_prom, pred_value = model.get_model_move_and_state(model_board)
    val_moves, _, action_probs = model.get_mcts_moves(cuda_board, pred_act, pred_prom, move_layer)
    to_contribs, from_contribs = get_to_and_from_move_contributions(val_moves[0], action_probs[0])

    drawer = BoardDrawer(draw_tensor, colour_list[0].cpu().item(), to_contribs, from_contribs)
    promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
    no_selected_move = torch.zeros((8, 8))
    new_game = True

    while new_game:
        batched_board = boards.to_tensor()
        colour_list = torch.Tensor([True]).to(torch.bool).cuda()
        cuda_board = batched_board.cuda()
        model_board = cuda_board[:, 0:6, :, :]
        draw_tensor = batched_board[0, 0:6, :, :]
        pred_act, pred_prom, pred_value = model.get_model_move_and_state(model_board)
        drawer.colour = colour_list[0].cpu().item()
        drawer.update_board(draw_tensor, no_selected_move, to_contribs, from_contribs)
        dud_move_count = boards.get_starting_move_count_list().cuda()
    
        current_move_select = no_selected_move
        previous_select_x = None
        previous_select_y = None
        while True:
            event, value = drawer.window.Read() # Position is the event
            if event == sg.WIN_CLOSED:
                new_game = False
                break
            move_layer = chess_cpp.get_moves_for_player(cuda_board)
            # Future - some sort of support for grouping over promotions (the second returned parameter) could be added
            val_moves, _, action_probs = model.get_mcts_moves(cuda_board, pred_act, pred_prom, move_layer)
            to_contribs, from_contribs = get_to_and_from_move_contributions(val_moves[0], action_probs[0])
            select_y, select_x = event
            select_y = 7 - select_y if drawer.colour else select_y
            select_x = select_x if drawer.colour else 7 - select_x
            flat_pos = select_y * 8 + select_x
            if select_y == previous_select_y and select_x == previous_select_x:
                current_move_select = no_selected_move
                drawer.update_board(draw_tensor, current_move_select, to_contribs, from_contribs)
            elif current_move_select[select_y][select_x] == 1:
                move = torch.Tensor([[previous_select_y, previous_select_x, select_y, select_x]]).to(torch.int8).cuda()
                is_promotion = chess_cpp.get_pawn_promote_move_mask(cuda_board, move)
                if is_promotion[0]:
                    promotion_gui = drawer.select_promotion_piece().to(torch.int8).cuda()
                else:
                    promotion_gui = promotion_nothing
                prev_board = torch.clone(batched_board)
                chess_cpp.enact_moves(cuda_board, move, promotion_gui, dud_move_count)
                flipped_board, colour_list = flip_board(cuda_board, colour_list)
                opponents_move_layer = chess_cpp.get_moves_for_player(flipped_board)
                white_view = get_white_view(cuda_board, flipped_board, torch.logical_not(colour_list))
                batched_board = flipped_board.cpu()
                repetition_status = get_repetition_status(boards, white_view)
                game_over = is_game_over(flipped_board, opponents_move_layer, repetition_status, dud_move_count)
                game_over_message = get_game_over_message(game_over[0], torch.logical_not(colour_list))
                draw_tensor = white_view.cpu()[0, 0:6, :, :]
                drawer.flip_turn()
                current_move_select = no_selected_move
                drawer.update_board(draw_tensor, no_selected_move, to_contribs, from_contribs)
                previous_select_x = None
                previous_select_y = None
                if game_over_message is not None:
                    choice = sg.popup_yes_no(f"{game_over_message}\n\nStart a new game?",  title="Game over")
                    if choice == "No":
                        new_game = False
                        break
                    elif choice == "Yes":
                        boards.reset_repetitions(torch.any(game_over).unsqueeze(0))
                        break
            else:
                previous_select_x = select_x
                previous_select_y = select_y
                this_ml = move_layer[0].cpu()
                current_move_select = this_ml[flat_pos].reshape((8, 8))
                drawer.update_board(draw_tensor, current_move_select, to_contribs, from_contribs)

    drawer.close()
