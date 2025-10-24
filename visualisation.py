# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:36:36 2023

Adapted heavily from Python-Easy-Chess-GUI:
    https://github.com/fsmosca/Python-Easy-Chess-GUI.git
    Original author: fsmosca (https://github.com/fsmosca)

Also from Chess 1.0's Visualisation script.

@author: jledragon
"""


import torch
from enum import Enum
import os
import PySimpleGUI as sg
import copy

IMAGE_PATH = 'PieceImages'  # path to the chess pieces
SQ_LIGHT_COLOUR = '#F0D9B5'
SQ_DARK_COLOUR = '#B58863'
MOVE_SQ_LIGHT_COLOUR = '#E8E18E'
MOVE_SQ_DARK_COLOUR = '#B8AF4E'

class PieceNames(Enum):
    EMPTY = 0
    BLACK_PAWN = 1
    BLACK_KNIGHT = 2
    BLACK_BISHOP = 3
    BLACK_ROOK = 4
    BLACK_KING = 5
    BLACK_QUEEN = 6
    WHITE_PAWN = 7
    WHITE_KNIGHT = 8
    WHITE_BISHOP = 9
    WHITE_ROOK = 10
    WHITE_KING = 11
    WHITE_QUEEN = 12

piece_files = {
    PieceNames.EMPTY: os.path.join(IMAGE_PATH, 'blank.png'),
    PieceNames.BLACK_PAWN: os.path.join(IMAGE_PATH, 'bP.png'),
    PieceNames.BLACK_KNIGHT: os.path.join(IMAGE_PATH, 'bN.png'),
    PieceNames.BLACK_BISHOP: os.path.join(IMAGE_PATH, 'bB.png'),
    PieceNames.BLACK_ROOK: os.path.join(IMAGE_PATH, 'bR.png'),
    PieceNames.BLACK_KING: os.path.join(IMAGE_PATH, 'bK.png'),
    PieceNames.BLACK_QUEEN: os.path.join(IMAGE_PATH, 'bQ.png'),
    PieceNames.WHITE_PAWN: os.path.join(IMAGE_PATH, 'wP.png'),
    PieceNames.WHITE_KNIGHT: os.path.join(IMAGE_PATH, 'wN.png'),
    PieceNames.WHITE_BISHOP: os.path.join(IMAGE_PATH, 'wB.png'),
    PieceNames.WHITE_ROOK: os.path.join(IMAGE_PATH, 'wR.png'),
    PieceNames.WHITE_KING: os.path.join(IMAGE_PATH, 'wK.png'),
    PieceNames.WHITE_QUEEN: os.path.join(IMAGE_PATH, 'wQ.png'),
}

white_init_promote_board = [[PieceNames.WHITE_QUEEN, PieceNames.WHITE_ROOK, PieceNames.WHITE_BISHOP, PieceNames.WHITE_KNIGHT]]
black_init_promote_board = [[PieceNames.BLACK_QUEEN, PieceNames.BLACK_ROOK, PieceNames.BLACK_BISHOP, PieceNames.BLACK_KNIGHT]]

promotables = {
    PieceNames.WHITE_QUEEN: torch.Tensor([0, 0, 0, 1]),
    PieceNames.WHITE_ROOK:  torch.Tensor([0, 0, 1, 0]),
    PieceNames.WHITE_BISHOP:  torch.Tensor([0, 1, 0, 0]),
    PieceNames.WHITE_KNIGHT:  torch.Tensor([1, 0, 0, 0]),
    PieceNames.BLACK_QUEEN:  torch.Tensor([0, 0, 0, 1]),
    PieceNames.BLACK_ROOK:  torch.Tensor([0, 0, 1, 0]),
    PieceNames.BLACK_BISHOP:  torch.Tensor([0, 1, 0, 0]),
    PieceNames.BLACK_KNIGHT:  torch.Tensor([1, 0, 0, 0]),
}

class BoardDrawer:
    '''
    A class used to create a chessboard GUI
    '''
    
    def __init__(self, board_tensor, starting_colour):
        '''
        Create the window. Set half a second delay between moves.
        '''
        self.colour = starting_colour
        self.board_layout = self.create_board(board_tensor)
        board_tab = [[sg.Column(self.board_layout)]]
        layout = [[sg.Column(board_tab)]]  
        self.window = sg.Window('A lovely game of chess.', layout)
        self.timeout = 500
        self.psg_board = None
        self.can_move = torch.zeros((8, 8))
        self.update_board(board_tensor, self.can_move)

    def update_board(self, board_tensor, move_tensor):
        '''
        Update the visuals of the board.
        '''
        event, values = self.window.read(timeout=self.timeout)
        self.can_move = move_tensor
        self.psg_board = self.convertBoard(board_tensor)
        self.redraw_board(self.window)

    def render_square(self, image, key, location):
        '''
        Render one square on the visual board.
        '''
        even = (location[0] + location[1]) % 2
        colour = SQ_DARK_COLOUR if even else SQ_LIGHT_COLOUR
        return sg.RButton('', image_filename=image, size=(1, 1),
                          border_width=0, button_color=('white', colour),
                          pad=(0, 0), key=key)

    def convertBoard(self, board_tensor):
        '''
        Convert board from JLE cpp board noration to something that can be drawn.
        The board will be the first batch index and the 0-5 encodings.
        '''
        converted = []
        pieces_no_col = board_tensor[0:5, :, :]
        piece_encoding_sum = torch.sum(pieces_no_col, 0)
        board_encoding_sum = torch.sum(board_tensor, 0)
        my_colour = 0
        for row in range(board_tensor.shape[1] - 1, -1, -1):
            conv_row = []
            for col in range(board_tensor.shape[2]):
                if board_encoding_sum[row][col] == 0:
                    conv_row.append(PieceNames.EMPTY)
                elif pieces_no_col[2][row][col] == 1 and pieces_no_col[3][row][col] == 1 and piece_encoding_sum[row][col] == 2:
                    if board_tensor[5][row][col] == my_colour:
                        conv_row.append(PieceNames.WHITE_QUEEN)
                    else:
                        conv_row.append(PieceNames.BLACK_QUEEN)
                elif pieces_no_col[3][row][col] == 1 and piece_encoding_sum[row][col] == 1:
                    if board_tensor[5][row][col] == my_colour:
                        conv_row.append(PieceNames.WHITE_ROOK)
                    else:
                        conv_row.append(PieceNames.BLACK_ROOK)
                elif pieces_no_col[2][row][col] == 1 and piece_encoding_sum[row][col] == 1:
                    if board_tensor[5][row][col] == my_colour:
                        conv_row.append(PieceNames.WHITE_BISHOP)
                    else:
                        conv_row.append(PieceNames.BLACK_BISHOP)
                elif pieces_no_col[1][row][col] == 1 and piece_encoding_sum[row][col] == 1:
                    if board_tensor[5][row][col] == my_colour:
                        conv_row.append(PieceNames.WHITE_KNIGHT)
                    else:
                        conv_row.append(PieceNames.BLACK_KNIGHT)
                elif pieces_no_col[0][row][col] == 1 and piece_encoding_sum[row][col] == 1:
                    if board_tensor[5][row][col] == my_colour:
                        conv_row.append(PieceNames.WHITE_PAWN)
                    else:
                        conv_row.append(PieceNames.BLACK_PAWN)
                elif pieces_no_col[4][row][col] == 1 and piece_encoding_sum[row][col] == 1:
                    if board_tensor[5][row][col] == my_colour:
                        conv_row.append(PieceNames.WHITE_KING)
                    else:
                        conv_row.append(PieceNames.BLACK_KING)
            converted.append(conv_row)
        return converted
    
    def flip_turn(self):
        '''
        Tell the drawer that the player taking a move has changed.
        '''
        self.colour = not self.colour
    
    def select_promotion_piece(self):
        '''
        Open the GUI showing the user which piece to promote their pawn to.
        '''
        piece = None
        board_layout, row = [], []

        psg_promote_board = copy.deepcopy(
            white_init_promote_board
        ) if self.colour else copy.deepcopy(
            black_init_promote_board
        )
            
        # Loop through board and create buttons with images        
        for j in range(4):
            piece_image = piece_files[psg_promote_board[0][j]]
            row.append(self.render_square(piece_image, key=(0, j),
                                          location=(0, j)))
        board_layout.append(row)
        
        promo_window = sg.Window('Which piece would you like?',
                                 board_layout,
                                 default_button_element_size=(12, 1),
                                 auto_size_buttons=False,
                                 icon='Icon/pecg.ico')
        
        button, value = promo_window.Read()
        if button is not None and type(button) is tuple:
            move_from = button
            fr_row, fr_col = move_from
            piece = psg_promote_board[fr_row][fr_col]

        promo_window.Close()

        return promotables[piece]

    def create_board(self, board_tensor):
        '''
        Create the board.
        '''
        self.psg_board = self.convertBoard(board_tensor)
        board_layout = []
    
        if self.colour:
            # Save the board with black at the top        
            start = 0
            end = 8
            step = 1
        else:
            start = 7
            end = -1
            step = -1
    
        # Loop through the board and create buttons with images
        for i in range(start, end, step):
            # Row numbers at left of board is blank
            row = []
            for j in range(start, end, step):
                piece_image = piece_files[self.psg_board[i][j]]
                row.append(self.render_square(piece_image, key=(i, j), location=(i, j)))
            board_layout.append(row)
    
        return board_layout

    def redraw_board(self, window):
        '''
        Redraw the board. This should be called after a move is made.
        '''
        for i in range(8):
            i_view = i if self.colour else 7 - i
            for j in range(8):
                j_view = j if self.colour else 7 - j
                if self.can_move[7 - i][j] == 1:
                    colour = MOVE_SQ_DARK_COLOUR if (i_view + j_view) % 2 else MOVE_SQ_LIGHT_COLOUR
                else:
                    colour = SQ_DARK_COLOUR if (i_view + j_view) % 2 else SQ_LIGHT_COLOUR
                piece_image = piece_files[self.psg_board[i_view][j_view]]
                elem = window[(i_view, j_view)]
                elem.Update(button_color=('white', colour), image_filename=piece_image,)

    def close(self):
        '''
        Close the window.
        '''
        self.window.close()
