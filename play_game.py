# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:38:12 2023

Currently human vs. human.

@author: jledragon
"""


import torch
import chess_cpp
from chess_py_utils import flip_board, is_game_over, get_white_view, get_repetition_status, get_game_over_message
from visualisation import BoardDrawer
import FreeSimpleGUI as sg
from pygame import mixer
import os

BATCH_SIZE = 1

def is_pawn_promote_move(batched_board, prev_x, prev_y, x, y, colour_list):
    # Deprecated - Use new C++ function.
    enc = batched_board[0, :, prev_y, prev_x]
    piece_enc_sum = torch.sum(enc[0:5])
    # If we're a pawn moving to the end of the board...
    if enc[0] == 1 and piece_enc_sum == 1 and y == 7:
        return True
    else:
        return False

def is_en_passant_move(batched_board, prev_x, prev_y, x, y):
    from_enc = batched_board[0, :, prev_y, prev_x]
    from_piece_enc = torch.sum(from_enc[0:5])
    to_enc_sum = torch.sum(batched_board[0, :, y, x])
    # If we're a pawn, moving diagonally to a square with nothing there...
    if from_enc[0] == 1 and from_piece_enc == 1 and abs(prev_x - x) == 1 and to_enc_sum == 0:
        return True
    else:
        return False

def is_taking_move(batched_board, x, y):
    to_enc_sum = torch.sum(batched_board[0, :, y, x])
    # If something is there at the square we're trying to move to...
    if to_enc_sum > 1:
        return True
    else:
        return False

class SoundMaker:
    def __init__(self):
        mixer.init()
        self._sound_dir = "sounds"
        self._game_over = os.path.join(self._sound_dir, "notify.mp3")
        self._move = os.path.join(self._sound_dir, "move-self.mp3")
        self._take = os.path.join(self._sound_dir, "capture.mp3")

    def make_sound(self, batched_board, prev_x, prev_y, x, y, is_game_over):
        if is_game_over:
            mixer.music.load(self._game_over)
        elif is_en_passant_move(batched_board, prev_x, prev_y, x, y):
            mixer.music.load(self._take)
        elif is_taking_move(batched_board, x, y):
            mixer.music.load(self._take)
        else:
            mixer.music.load(self._move)
        mixer.music.play()

if __name__ == '__main__':
    assert BATCH_SIZE == 1, "Only a single batch may be visualised."

    new_game = True
    starting_boards = chess_cpp.BatchedBoard(True, BATCH_SIZE, 0)
    batched_board = starting_boards.to_tensor()
    chess_cpp.get_moves_for_player(batched_board.cuda()) # Warm up the GPU for real-time speed.
    draw_tensor = batched_board[0, 0:6, :, :]
    colour_list = torch.Tensor([True]).to(torch.bool).cuda()
    drawer = BoardDrawer(draw_tensor, colour_list[0].cpu().item())
    promotion_nothing = torch.Tensor([[0, 0, 0, 0]]).to(torch.int8).cuda()
    no_selected_move = torch.zeros((8, 8))
    sound_maker = SoundMaker()

    while new_game:
        batched_board = starting_boards.to_tensor()
        colour_list = torch.Tensor([True]).to(torch.bool).cuda()
        draw_tensor = batched_board[0, 0:6, :, :]
        drawer.colour = colour_list[0].cpu().item()
        drawer.update_board(draw_tensor, no_selected_move)
        dud_move_count = starting_boards.get_starting_move_count_list().cuda()
    
        current_move_select = no_selected_move
        previous_select_x = None
        previous_select_y = None
        while True:
            event, value = drawer.window.Read() # Position is the event
            if event == sg.WIN_CLOSED:
                new_game = False
                break
            move_layer = chess_cpp.get_moves_for_player(batched_board.cuda())[0].cpu()
            select_y, select_x = event
            select_y = 7 - select_y if drawer.colour else select_y
            select_x = select_x if drawer.colour else 7 - select_x
            flat_pos = select_y * 8 + select_x
            if select_y == previous_select_y and select_x == previous_select_x:
                current_move_select = no_selected_move
                drawer.update_board(draw_tensor, current_move_select)
            elif current_move_select[select_y][select_x] == 1:
                move = torch.Tensor([[previous_select_y, previous_select_x, select_y, select_x]]).to(torch.int8).cuda()
                is_pawn_promote = is_pawn_promote_move(batched_board, previous_select_x, previous_select_y, select_x, select_y, colour_list)
                if is_pawn_promote:
                    promotion_gui = drawer.select_promotion_piece().to(torch.int8).cuda()
                else:
                    promotion_gui = promotion_nothing
                prev_board = torch.clone(batched_board)
                cuda_board = batched_board.cuda()
                chess_cpp.enact_moves(cuda_board, move, promotion_gui, dud_move_count)
                flipped_board, colour_list = flip_board(cuda_board, colour_list)
                opponents_move_layer = chess_cpp.get_moves_for_player(flipped_board)
                white_view = get_white_view(cuda_board, flipped_board, torch.logical_not(colour_list))
                batched_board = flipped_board.cpu()
                repetition_status = get_repetition_status(starting_boards, white_view)
                game_over = is_game_over(flipped_board, opponents_move_layer, repetition_status, dud_move_count)
                game_over_message = get_game_over_message(game_over[0], torch.logical_not(colour_list))
                draw_tensor = white_view.cpu()[0, 0:6, :, :]
                drawer.flip_turn()
                current_move_select = no_selected_move
                drawer.update_board(draw_tensor, no_selected_move)
                sound_maker.make_sound(prev_board, previous_select_x, previous_select_y, select_x, select_y, torch.any(game_over))
                previous_select_x = None
                previous_select_y = None
                if game_over_message is not None:
                    choice = sg.popup_yes_no(f"{game_over_message}\n\nStart a new game?",  title="Game over")
                    if choice == "No":
                        new_game = False
                        break
                    elif choice == "Yes":
                        starting_boards.reset_repetitions(torch.any(game_over).unsqueeze(0))
                        break
            else:
                previous_select_x = select_x
                previous_select_y = select_y
                current_move_select = move_layer[flat_pos].reshape((8, 8))
                drawer.update_board(draw_tensor, current_move_select)

    drawer.close()
