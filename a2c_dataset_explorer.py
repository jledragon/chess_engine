# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:45:57 2026

@author: jledragon

This script will open an interactive chessboard GUI with some starting position and with some loaded dataset.
When opened, the first position in the dataset is loaded with 'from' pieces highlighted in blue and 'to' pieces
highlighted in red, similar to how the model explorer works, using the saved mcts_prob_buffer tensor. On
selecting a piece, it's 'to' positions will also be highlighted in red. Making moves will not be possible in this
GUI. The saved value's mean and standard deviation, as well as number of times seen will be displayed underneath
the board. There will also be a 'next' and 'previous' button to move onto the next and previous saved position
in the dataset. This will help us to debug the MCTS process.

"""

