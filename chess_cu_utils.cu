#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


template<typename scalar_t> __device__ bool anyPieceHere(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos) {
    int square = yPos * boardSize + xPos;
    int pawnEncodingHere = blockIdx.x * blockDim.x * encodingSize + square;
    int knightEncodingHere = blockIdx.x * blockDim.x * encodingSize + square + blockDim.x;
    int bishopEncodingHere = blockIdx.x * blockDim.x * encodingSize + square + 2 * blockDim.x;
    int rookEncodingHere = blockIdx.x * blockDim.x * encodingSize + square + 3 * blockDim.x;
    int kingEncodingHere = blockIdx.x * blockDim.x * encodingSize + square + 4 * blockDim.x;
    if (board[pawnEncodingHere] == 1 || board[knightEncodingHere] == 1 || board[bishopEncodingHere] == 1 || board[rookEncodingHere] == 1 || board[kingEncodingHere] == 1) {
        return true;
    }
    else {
        return false;
    }
}

template<typename scalar_t> __device__ bool enemyPieceHere(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos) {
    int square = yPos * boardSize + xPos;
    bool pieceHere = anyPieceHere(board, boardSize, encodingSize, xPos, yPos);
    int colourOfPiece = blockIdx.x * blockDim.x * encodingSize + square + 5 * blockDim.x;
    bool enemyPiece = (board[colourOfPiece] == 1);
    if (pieceHere && enemyPiece) {
        return true;
    }
    else {
        return false;
    }
}

template<typename scalar_t> __device__ bool friendlyPieceHere(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos) {
    int square = yPos * boardSize + xPos;
    bool pieceHere = anyPieceHere(board, boardSize, encodingSize, xPos, yPos);
    int colourOfPiece = blockIdx.x * blockDim.x * encodingSize + square + 5 * blockDim.x;
    bool friendlyPiece = (board[colourOfPiece] == 0);
    if (pieceHere && friendlyPiece) {
        return true;
    }
    else {
        return false;
    }
}

template<typename scalar_t> __device__ bool isSquareInPawnCheck(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos) {
    int thisBatch = blockIdx.x * blockDim.x * encodingSize;

    if (yPos + 1 < boardSize && xPos + 1 < boardSize) {
        int pawn1 = (yPos + 1) * boardSize + (xPos + 1);

        int pawnEncodingHere = thisBatch + pawn1;
        int colourOfPiece = thisBatch + pawn1 + 5 * blockDim.x;
        if (board[pawnEncodingHere] == 1 && board[colourOfPiece] == 1) {
            return true;
        }
    }
    if (yPos + 1 < boardSize && xPos - 1 >= 0) {
        int pawn2 = (yPos + 1) * boardSize + (xPos - 1);

        int pawnEncodingHere = thisBatch + pawn2;
        int colourOfPiece = thisBatch + pawn2 + 5 * blockDim.x;
        if (board[pawnEncodingHere] == 1 && board[colourOfPiece] == 1) {
            return true;
        }
    }
    
    return false;
}

template<typename scalar_t> __device__ bool isSquareInKnightCheck(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos) {
    int sweep_A[2] = {-2, 2};
    int sweep_B[2] = {-1, 1};
    int flatPos = yPos * boardSize + xPos;
    int thisBatch = blockIdx.x * blockDim.x * encodingSize;

    for (int swA=0; swA<2; swA++) {
        for (int swB=0; swB<2; swB++) {
            int move1x = sweep_B[swB];
            int move1y = sweep_A[swA] * boardSize;
            int move2x = sweep_A[swA];
            int move2y = sweep_B[swB] * boardSize;
            if (xPos + move1x >= 0 && xPos + move1x < boardSize && yPos + sweep_A[swA] >= 0 && yPos + sweep_A[swA] < boardSize) {
                int knight1 = move1y + move1x + flatPos;
                
                int knightEncodingHere = thisBatch + knight1 + blockDim.x;
                int colourOfPiece = thisBatch + knight1 + 5 * blockDim.x;
                if (board[knightEncodingHere] == 1 && board[colourOfPiece] == 1) {
                    return true;
                }
            }
            if (xPos + move2x >= 0 && xPos + move2x < boardSize && yPos + sweep_B[swB] >= 0 && yPos + sweep_B[swB] < boardSize) {
                int knight2 = move2y + move2x + flatPos;
                
                int knightEncodingHere = thisBatch + knight2 + blockDim.x;
                int colourOfPiece = thisBatch + knight2 + 5 * blockDim.x;
                if (board[knightEncodingHere] == 1 && board[colourOfPiece] == 1) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

template<typename scalar_t> __device__ bool isSquareInBishopCheck(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos, int ignoreX=-1, int ignoreY=-1) {
    int dirs[2] = {-1, 1};
    int thisBatch = blockIdx.x * blockDim.x * encodingSize;

    for (int vertDir=0; vertDir<2; vertDir++) {
        for (int horizDir=0; horizDir<2; horizDir++) {
        
            int vDir = dirs[vertDir];
            int hDir = dirs[horizDir];
                
            for (int diag=1; diag<boardSize; diag++) {
                if (xPos + hDir * diag < boardSize && yPos + vDir * diag < boardSize && xPos + hDir * diag >= 0 && yPos + vDir * diag >= 0) {
                    if (anyPieceHere(board, boardSize, encodingSize, xPos + hDir * diag, yPos + vDir * diag)) {
                        if (xPos + hDir * diag == ignoreX && yPos + vDir * diag == ignoreY) {
                            continue;
                        }
                        int bishop = (yPos + vDir * diag) * boardSize + (xPos + hDir * diag);
                
                        int bishopEncodingHere = thisBatch + bishop + 2 * blockDim.x;
                        int colourOfPiece = thisBatch + bishop + 5 * blockDim.x;
                        if (board[bishopEncodingHere] == 1 && board[colourOfPiece] == 1) {
                            return true;
                        }
                        else {
                            break;
                        }
                    }
                }
                else {
                    break;
                }
            }
        }
    }
    
    return false;
}

template<typename scalar_t> __device__ bool isSquareInRookCheck(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos, int ignoreX=-1, int ignoreY=-1) {
    int dirs[2] = {-1, 1};
    int thisBatch = blockIdx.x * blockDim.x * encodingSize;

    for (int direction=0; direction<2; direction++) {
    
        int dir = dirs[direction];

        for (int rowCol=1; rowCol<boardSize; rowCol++) {
            if (xPos + dir * rowCol < boardSize && xPos + dir * rowCol >= 0) {
                if (anyPieceHere(board, boardSize, encodingSize, xPos + dir * rowCol, yPos)) {
                    if (xPos + dir * rowCol == ignoreX && yPos == ignoreY) {
                        continue;
                    }
                    int rook1 = yPos * boardSize + (xPos + dir * rowCol);
                    
                    int rookEncodingHere = thisBatch + rook1 + 3 * blockDim.x;
                    int colourOfPiece = thisBatch + rook1 + 5 * blockDim.x;
                    if (board[rookEncodingHere] == 1 && board[colourOfPiece] == 1) {
                        return true;
                    }
                    else {
                        break;
                    }
                }
            }
            else {
                break;
            }
        }
        for (int rowCol=1; rowCol<boardSize; rowCol++) {
            if (yPos + dir * rowCol < boardSize && yPos + dir * rowCol >= 0) {
                if (anyPieceHere(board, boardSize, encodingSize, xPos, yPos + dir * rowCol)) {
                    if (xPos == ignoreX && yPos + dir * rowCol == ignoreY) {
                        continue;
                    }
                    int rook2 = (yPos + dir * rowCol) * boardSize + xPos;
                    
                    int rookEncodingHere = thisBatch + rook2 + 3 * blockDim.x;
                    int colourOfPiece = thisBatch + rook2 + 5 * blockDim.x;
                    if (board[rookEncodingHere] == 1 && board[colourOfPiece] == 1) {
                        return true;
                    }
                    else {
                        break;
                    }
                }
            }
            else {
                break;
            }
        }
    }
    
    return false;
}

template<typename scalar_t> __device__ bool isSquareInKingCheck(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos) {
    int thisBatch = blockIdx.x * blockDim.x * encodingSize;
    
    for (int row=-1; row<2; row++) {
        for (int col=-1; col<2; col++) {
            if (row == 0 && col == 0) {
                continue;
            }
            if (xPos + row >= 0 && xPos + row < boardSize && yPos + col >= 0 && yPos + col < boardSize) {
                int king = (yPos + col) * boardSize + (xPos + row);
                int kingEncodingHere = thisBatch + king + 4 * blockDim.x;
                int colourOfPiece = thisBatch + king + 5 * blockDim.x;
                if (board[kingEncodingHere] == 1 && board[colourOfPiece] == 1) {
                    return true;
                }
            }
        }
    }

    return false;
}

template<typename scalar_t> __device__ bool* getBlockSquares(
    scalar_t* __restrict__ board,
    const int boardSize,
    const int encodingSize,
    int xPos,
    int yPos,
    bool* __restrict__ noBlockingMoves,
    bool* __restrict__ notCheck,
    bool* __restrict__ allowedBlockingMoves
) {
    for (int i=0; i<64; i++) {
        notCheck[i] = true;
    }
    int thisBatch = blockIdx.x * blockDim.x * encodingSize;
    
    // Find the King of my colour.
    bool kingFound = false;
    int kingPos = -1;
    for (int row=0; row<boardSize; row++) {
        for (int col=0; col<boardSize; col++) {
            int pos = row*boardSize + col;
            int kingEncodingHere = thisBatch + pos + 4 * blockDim.x;
            int colourOfPiece = thisBatch + pos + 5 * blockDim.x;
            if (board[kingEncodingHere] == 1 && board[colourOfPiece] == 0) {
                kingFound = true;
                kingPos = pos;
                break;
            }
        }
        if (kingFound) {
            break;
        }
    }
    if (!kingFound) {
        return notCheck; // No King, no problem.
    }
    
    int kingXPos = kingPos % boardSize;
    int kingYPos = kingPos / boardSize;
    bool alreadyCheck = false;
    int dirs[2] = {-1, 1};
    
    // Check pawn check
    if (kingYPos + 1 < boardSize && kingXPos + 1 < boardSize) {
        int pawn1 = (kingYPos + 1) * boardSize + (kingXPos + 1);

        int pawnEncodingHere = thisBatch + pawn1;
        int colourOfPiece = thisBatch + pawn1 + 5 * blockDim.x;
        if (board[pawnEncodingHere] == 1 && board[colourOfPiece] == 1) {
            if (alreadyCheck) {
                return noBlockingMoves; // Not possible yet, but to be consistent.
            }
            else {
                alreadyCheck = true;
                allowedBlockingMoves[pawn1] = true;
            }
        }
    }
    if (kingYPos + 1 < boardSize && kingXPos - 1 >= 0) {
        int pawn2 = (kingYPos + 1) * boardSize + (kingXPos - 1);

        int pawnEncodingHere = thisBatch + pawn2;
        int colourOfPiece = thisBatch + pawn2 + 5 * blockDim.x;
        if (board[pawnEncodingHere] == 1 && board[colourOfPiece] == 1) {
            if (alreadyCheck) {
                return noBlockingMoves;
            }
            else {
                alreadyCheck = true;
                allowedBlockingMoves[pawn2] = true;
            }
        }
    }
    
    // Check knight check
    int sweep_A[2] = {-2, 2};
    int sweep_B[2] = {-1, 1};
    for (int swA=0; swA<2; swA++) {
        for (int swB=0; swB<2; swB++) {
            int move1x = sweep_B[swB];
            int move1y = sweep_A[swA] * boardSize;
            int move2x = sweep_A[swA];
            int move2y = sweep_B[swB] * boardSize;
            if (kingXPos + move1x >= 0 && kingXPos + move1x < boardSize && kingYPos + sweep_A[swA] >= 0 && kingYPos + sweep_A[swA] < boardSize) {
                int knight1 = move1y + move1x + kingPos;
                
                int knightEncodingHere = thisBatch + knight1 + blockDim.x;
                int colourOfPiece = thisBatch + knight1 + 5 * blockDim.x;
                if (board[knightEncodingHere] == 1 && board[colourOfPiece] == 1) {
                    if (alreadyCheck) {
                        return noBlockingMoves;
                    }
                    else {
                        alreadyCheck = true;
                        allowedBlockingMoves[knight1] = true;
                    }
                }
            }
            if (kingXPos + move2x >= 0 && kingXPos + move2x < boardSize && kingYPos + sweep_B[swB] >= 0 && kingYPos + sweep_B[swB] < boardSize) {
                int knight2 = move2y + move2x + kingPos;
                
                int knightEncodingHere = thisBatch + knight2 + blockDim.x;
                int colourOfPiece = thisBatch + knight2 + 5 * blockDim.x;
                if (board[knightEncodingHere] == 1 && board[colourOfPiece] == 1) {
                    if (alreadyCheck) {
                        return noBlockingMoves;
                    }
                    else {
                        alreadyCheck = true;
                        allowedBlockingMoves[knight2] = true;
                    }
                }
            }
        }
    }
    
    // Check King check
    for (int row=-1; row<2; row++) {
        for (int col=-1; col<2; col++) {
            if (row == 0 && col == 0) {
                continue;
            }
            if (kingXPos + row >= 0 && kingXPos + row < boardSize && kingYPos + col >= 0 && kingYPos + col < boardSize) {
                int king = (kingYPos + col) * boardSize + (kingXPos + row);
                int kingEncodingHere = thisBatch + king + 4 * blockDim.x;
                int colourOfPiece = thisBatch + king + 5 * blockDim.x;
                if (board[kingEncodingHere] == 1 && board[colourOfPiece] == 1) {
                    if (alreadyCheck) {
                        return noBlockingMoves;
                    }
                    else {
                        alreadyCheck = true;
                        allowedBlockingMoves[king] = true;
                    }
                }
            }
        }
    }
    
    // Check bishop check - careful, double check with  bishop/rook/queen should be unblockable.
    for (int vertDir=0; vertDir<2; vertDir++) {
        for (int horizDir=0; horizDir<2; horizDir++) {
        
            int vDir = dirs[vertDir];
            int hDir = dirs[horizDir];

            for (int diag=1; diag<boardSize; diag++) {
                if (kingXPos + hDir * diag < boardSize && kingYPos + vDir * diag < boardSize && kingXPos + hDir * diag >= 0 && kingYPos + vDir * diag >= 0) {
                    if (kingXPos + hDir * diag == xPos && kingYPos + vDir * diag == yPos) {
                        continue;
                    }
                    if (anyPieceHere(board, boardSize, encodingSize, kingXPos + hDir * diag, kingYPos + vDir * diag)) {
                        int bishop = (kingYPos + vDir * diag) * boardSize + (kingXPos + hDir * diag);

                        int bishopEncodingHere = thisBatch + bishop + 2 * blockDim.x;
                        int colourOfPiece = thisBatch + bishop + 5 * blockDim.x;
                        if (board[bishopEncodingHere] == 1 && board[colourOfPiece] == 1) {
                            if (alreadyCheck) {
                                return noBlockingMoves; // Double check - no blocking moves.
                            }
                            else {
                                alreadyCheck = true;
                                for (int backdiag=diag; backdiag>0; backdiag--) {
                                    int block1 = (kingYPos + vDir * backdiag) * boardSize + (kingXPos + hDir * backdiag);
                                    allowedBlockingMoves[block1] = true;
                                }
                            }
                        }
                        else {
                            break;
                        }
                    }
                }
                else {
                    break;
                }
            }
        }
    }
    
    // Check rook check - careful, double check with  bishop/rook/queen should be unblockable.
    for (int direction=0; direction<2; direction++) {
        int dir = dirs[direction];
        
        for (int rowCol=1; rowCol<boardSize; rowCol++) {
            if (kingXPos + dir * rowCol < boardSize && kingXPos + dir * rowCol >= 0) {
                if (kingXPos + dir * rowCol == xPos && kingYPos == yPos) {
                    continue;
                }
                if (anyPieceHere(board, boardSize, encodingSize, kingXPos + dir * rowCol, kingYPos)) {
                    int rook1 = kingYPos * boardSize + (kingXPos + dir * rowCol);
                    
                    int rookEncodingHere = thisBatch + rook1 + 3 * blockDim.x;
                    int colourOfPiece = thisBatch + rook1 + 5 * blockDim.x;
                    if (board[rookEncodingHere] == 1 && board[colourOfPiece] == 1) {
                        if (alreadyCheck) {
                            return noBlockingMoves; // Double check - no blocking moves.
                        }
                        else{
                            alreadyCheck = true;
                            for (int backrowCol=rowCol; backrowCol>0; backrowCol--) {
                                int block1 = kingYPos * boardSize + (kingXPos + dir * backrowCol);
                                allowedBlockingMoves[block1] = true;
                            }
                            break;
                        }
                    }
                    else {
                        break;
                    }
                }
            }
            else {
                break;
            }
        }
        for (int rowCol=1; rowCol<boardSize; rowCol++) {
            if (kingYPos + dir * rowCol < boardSize && kingYPos + dir * rowCol >= 0) {
                if (kingXPos == xPos && kingYPos + dir * rowCol == yPos) {
                    continue;
                }
                if (anyPieceHere(board, boardSize, encodingSize, kingXPos, kingYPos + dir * rowCol)) {
                    int rook2 = (kingYPos + dir * rowCol) * boardSize + kingXPos;
                    
                    int rookEncodingHere = thisBatch + rook2 + 3 * blockDim.x;
                    int colourOfPiece = thisBatch + rook2 + 5 * blockDim.x;
                    if (board[rookEncodingHere] == 1 && board[colourOfPiece] == 1) {
                        if (alreadyCheck) {
                            return noBlockingMoves; // Double check - no blocking moves.
                        }
                        else{
                            alreadyCheck = true;
                            for (int backrowCol=rowCol; backrowCol>0; backrowCol--) {
                                int block2 = (kingYPos + dir * backrowCol) * boardSize + kingXPos;
                                allowedBlockingMoves[block2] = true;
                            }
                            break;
                        }
                    }
                    else {
                        break;
                    }
                }
            }
            else {
                break;
            }
        }
    }
    
    if (alreadyCheck) {
        return allowedBlockingMoves;
    }
    else{
        return notCheck;
    }
}

template<typename scalar_t> __device__ bool* getAllowedSquaresGivenPossiblyPinned(scalar_t* __restrict__ board) {
    bool placeholder[64];
    return placeholder;
}

template<typename scalar_t> __device__ bool isSquareInCheck(scalar_t* __restrict__ board, const int boardSize, const int encodingSize, int xPos, int yPos, int ignoreX=-1, int ignoreY=-1) {
    if (isSquareInPawnCheck(board, boardSize, encodingSize, xPos, yPos)) {
        return true;
    }
    else if (isSquareInKnightCheck(board, boardSize, encodingSize, xPos, yPos)) {
        return true;
    }
    else if (isSquareInBishopCheck(board, boardSize, encodingSize, xPos, yPos, ignoreX, ignoreY)) {
        return true;
    }
    else if (isSquareInRookCheck(board, boardSize, encodingSize, xPos, yPos, ignoreX, ignoreY)) {
        return true;
    }
    else if (isSquareInKingCheck(board, boardSize, encodingSize, xPos, yPos)) {
        return true;
    }
    else {
        return false;
    }
}

template<typename scalar_t> __device__ void getMoves_device_cu(
    scalar_t* __restrict__ board,
    scalar_t* __restrict__ moveLayer,
    bool* __restrict__ nothingAllowed,
    bool* __restrict__ everythingAllowed,
    bool* __restrict__ validBlockingSquares,
    const int boardSize,
    const int encodingSize
) {
    int moveFrom = (blockIdx.x * blockDim.x + threadIdx.x) * boardSize * boardSize;
    
    // Checks for pieces on this square.
    int pawnEncodingHere = blockIdx.x * blockDim.x * encodingSize + threadIdx.x;
    int knightEncodingHere = blockIdx.x * blockDim.x * encodingSize + threadIdx.x + blockDim.x;
    int bishopEncodingHere = blockIdx.x * blockDim.x * encodingSize + threadIdx.x + 2 * blockDim.x;
    int rookEncodingHere = blockIdx.x * blockDim.x * encodingSize + threadIdx.x + 3 * blockDim.x;
    int kingEncodingHere = blockIdx.x * blockDim.x * encodingSize + threadIdx.x + 4 * blockDim.x;
    int colourOfPiece = blockIdx.x * blockDim.x * encodingSize + threadIdx.x + 5 * blockDim.x;
    int isFirstMoveHere = blockIdx.x * blockDim.x * encodingSize + threadIdx.x + 6 * blockDim.x;
    bool myPiece = (board[colourOfPiece] == 0);
    int xPos = threadIdx.x % boardSize;
    int yPos = threadIdx.x / boardSize;

    bool* notAllowedPos = &nothingAllowed[moveFrom];
    bool* allAllowedPos = &everythingAllowed[moveFrom];
    bool* validPos = &validBlockingSquares[moveFrom];
    bool* allowed = getBlockSquares(board, boardSize, encodingSize, xPos, yPos, notAllowedPos, allAllowedPos, validPos);
    
    if ((board[pawnEncodingHere] == 1) && myPiece) {
        // Deal with pawn moves.
        if (board[isFirstMoveHere] == 1) {
            int moveTwo = threadIdx.x + 2 * boardSize;
            if (moveTwo < boardSize * boardSize) { // Should never happen, but for sanity.
                if (allowed[moveTwo] && !anyPieceHere(board, boardSize, encodingSize, xPos, yPos + 1) && !anyPieceHere(board, boardSize, encodingSize, xPos, yPos + 2)) {
                    moveLayer[moveFrom + moveTwo] = 1; // Two forwards.
                }
            }
        }
		int moveOne = threadIdx.x + boardSize;
        if (moveOne < boardSize * boardSize) { // Should never happen, but for sanity.
            if (allowed[moveOne] && !anyPieceHere(board, boardSize, encodingSize, xPos, yPos + 1)) {
                moveLayer[moveFrom + moveOne] = 1; // One forwards.
            }
        }
        
        int takeOne = (yPos + 1) * boardSize + (xPos + 1);
        if (takeOne < boardSize * boardSize && enemyPieceHere(board, boardSize, encodingSize, xPos + 1, yPos + 1) && xPos < boardSize - 1) {
            if (allowed[takeOne]) {
                moveLayer[moveFrom + takeOne] = 1;
            }
        }
        int takeTwo = (yPos + 1) * boardSize + (xPos - 1);
        if (takeOne < boardSize * boardSize && enemyPieceHere(board, boardSize, encodingSize, xPos - 1, yPos + 1) && xPos > 0) {
            if (allowed[takeTwo]) {
                moveLayer[moveFrom + takeTwo] = 1;
            }
        }

        // En passant moves
        if (xPos + 1 < boardSize && yPos + 1 < boardSize) {
            int enPassantRight = yPos * boardSize + (xPos + 1);
            bool enemyPresent = enemyPieceHere(board, boardSize, encodingSize, xPos + 1, yPos);
            int pawnTarget = blockIdx.x * blockDim.x * encodingSize + enPassantRight;
            int justTookFirstMove = blockIdx.x * blockDim.x * encodingSize + enPassantRight + 7 * blockDim.x;
            if (board[justTookFirstMove] == 1 && board[pawnTarget] == 1 && enemyPresent) {
                int ep1 = (yPos + 1) * boardSize + (xPos + 1);
                if (allowed[ep1]) {
                    moveLayer[moveFrom + ep1] = 1;
                }
            }
        }
        if (xPos - 1 >= 0 && yPos + 1 < boardSize) {
            int enPassantLeft = yPos * boardSize + (xPos - 1);
            bool enemyPresent = enemyPieceHere(board, boardSize, encodingSize, xPos - 1, yPos);
            int pawnTarget = blockIdx.x * blockDim.x * encodingSize + enPassantLeft;
            int justTookFirstMove = blockIdx.x * blockDim.x * encodingSize + enPassantLeft + 7 * blockDim.x;
            if (board[justTookFirstMove] == 1 && board[pawnTarget] == 1 && enemyPresent) {
                int ep2 = (yPos + 1) * boardSize + (xPos - 1);
                if (allowed[ep2]) {
                    moveLayer[moveFrom + ep2] = 1;
                }
            }
        }
    }
    
    if ((board[knightEncodingHere] == 1) && myPiece) {
        // Deal with knight moves.
        
        // Capture the L-shaped movement of a knight.
        int sweep_A[2] = {-2, 2};
        int sweep_B[2] = {-1, 1};
        // -2:-1, -2:1, 2:-1, 2:1, plus all in reverse
        for (int swA=0; swA<2; swA++) {
            for (int swB=0; swB<2; swB++) {
                int move1x = sweep_B[swB];
                int move1y = sweep_A[swA] * boardSize;
                int move2x = sweep_A[swA];
                int move2y = sweep_B[swB] * boardSize;
                int move1 = move1y + move1x + threadIdx.x;
                int move2 = move2y + move2x + threadIdx.x;
                if (xPos + move1x >= 0 && xPos + move1x < boardSize && yPos + sweep_A[swA] >= 0 && yPos + sweep_A[swA] < boardSize) {
                    if (!friendlyPieceHere(board, boardSize, encodingSize, xPos + move1x, yPos + sweep_A[swA])) {
                        if (allowed[move1]) {
                            moveLayer[moveFrom + move1] = 1;
                        }
                    }
                }
                if (xPos + move2x >= 0 && xPos + move2x < boardSize && yPos + sweep_B[swB] >= 0 && yPos + sweep_B[swB] < boardSize) {
                    if (!friendlyPieceHere(board, boardSize, encodingSize, xPos + move2x, yPos + sweep_B[swB])) {
                        if (allowed[move2]) {
                            moveLayer[moveFrom + move2] = 1;
                        }
                    }
                }
            }
        }
    }
    
    int dirs[2] = {-1, 1};
    // Queens incorporated into bishop + rook moves.
    if ((board[bishopEncodingHere] == 1) && myPiece) {
        // Deal with bishop moves.
        
        for (int vertDir=0; vertDir<2; vertDir++) {
            for (int horizDir=0; horizDir<2; horizDir++) {
            
                int vDir = dirs[vertDir];
                int hDir = dirs[horizDir];
    
                for (int diag=1; diag<boardSize; diag++) {
                    if (xPos + hDir * diag < boardSize && yPos + vDir * diag < boardSize && xPos + hDir * diag >= 0 && yPos + vDir * diag >= 0) {
                        if (friendlyPieceHere(board, boardSize, encodingSize, xPos + hDir * diag, yPos + vDir * diag)) {
                            break;
                        }
                        int move = yPos * boardSize + xPos + hDir * diag + boardSize * vDir * diag;
                        if (allowed[move]) {
                            moveLayer[moveFrom + move] = 1;
                        }
                        if (enemyPieceHere(board, boardSize, encodingSize, xPos + hDir * diag, yPos + vDir * diag)) {
                            break;
                        }
                    }
                }
            }
        }
    }
    
    if ((board[rookEncodingHere] == 1) && myPiece) {
        // Deal with rook moves.
        
        for (int direction=0; direction<2; direction++) {

            int dir = dirs[direction];

            for (int rowCol=1; rowCol<boardSize; rowCol++) {
                if (xPos + dir * rowCol < boardSize && xPos + dir * rowCol >= 0) {
                    if (friendlyPieceHere(board, boardSize, encodingSize, xPos + dir * rowCol, yPos)) {
                        break;
                    }
                    int move = yPos * boardSize + xPos + dir * rowCol;
                    if (allowed[move]) {
                        moveLayer[moveFrom + move] = 1;
                    }
                    if (enemyPieceHere(board, boardSize, encodingSize, xPos + dir * rowCol, yPos)) {
                        break;
                    }
                }
            }
            for (int rowCol=1; rowCol<boardSize; rowCol++) {
                if (yPos + dir * rowCol < boardSize && yPos + dir * rowCol >= 0) {
                    if (friendlyPieceHere(board, boardSize, encodingSize, xPos, yPos + dir * rowCol)) {
                        break;
                    }
                    int move = yPos * boardSize + xPos + dir * rowCol * boardSize;
                    if (allowed[move]) {
                        moveLayer[moveFrom + move] = 1;
                    }
                    if (enemyPieceHere(board, boardSize, encodingSize, xPos, yPos + dir * rowCol)) {
                        break;
                    }
                }
            }
        }
    }
    
    if ((board[kingEncodingHere] == 1) && myPiece) {
        // Deal with king moves.
        
        for (int row=-1; row<2; row++) {
            for (int col=-1; col<2; col++) {
                if (row == 0 && col == 0) {
                    continue;
                }
                if (xPos + row >= 0 && xPos + row < boardSize && yPos + col >= 0 && yPos + col < boardSize) {
                    if (!isSquareInCheck(board, boardSize, encodingSize, xPos + row, yPos + col, xPos, yPos)) {
                        if (friendlyPieceHere(board, boardSize, encodingSize, xPos + row, yPos + col)) {
                            continue;
                        }
                        int move = yPos * boardSize + xPos + row + boardSize * col;
                        moveLayer[moveFrom + move] = 1;
                    }
                }
            }
        }

        // Castling
        if (board[isFirstMoveHere] == 1 && xPos > 2 && xPos < boardSize - 2) {
            int queenRookTarget = yPos * boardSize;
            int rookEncodingQ = blockIdx.x * blockDim.x * encodingSize + queenRookTarget + 3 * blockDim.x;
            int bishopEncodingQ = blockIdx.x * blockDim.x * encodingSize + queenRookTarget + 2 * blockDim.x;
            int colourEncodingQ = blockIdx.x * blockDim.x * encodingSize + queenRookTarget + 5 * blockDim.x;
            int firstMoveQ = blockIdx.x * blockDim.x * encodingSize + queenRookTarget + 6 * blockDim.x;
            if (board[rookEncodingQ] == 1 && board[colourEncodingQ] == 0 && board[firstMoveQ] == 1 && board[bishopEncodingQ] == 0) {
                bool anythingBetween = false;
                for (int betweenSpace=1; betweenSpace<xPos; betweenSpace++) {
                    if (anyPieceHere(board, boardSize, encodingSize, betweenSpace, yPos)) {
                        anythingBetween = true;
                    }
                }
                if (!anythingBetween) {
                    if (!isSquareInCheck(board, boardSize, encodingSize, xPos, yPos) &&
                        !isSquareInCheck(board, boardSize, encodingSize, xPos - 1, yPos) &&
                        !isSquareInCheck(board, boardSize, encodingSize, xPos - 2, yPos)) {
                        int move = yPos * boardSize + xPos - 2;
                        moveLayer[moveFrom + move] = 1;
                    }
                }
            }

            int kingRookTarget = yPos * boardSize + (boardSize - 1);
            int rookEncodingK = blockIdx.x * blockDim.x * encodingSize + kingRookTarget + 3 * blockDim.x;
            int bishopEncodingK = blockIdx.x * blockDim.x * encodingSize + kingRookTarget + 2 * blockDim.x;
            int colourEncodingK = blockIdx.x * blockDim.x * encodingSize + kingRookTarget + 5 * blockDim.x;
            int firstMoveK = blockIdx.x * blockDim.x * encodingSize + kingRookTarget + 6 * blockDim.x;
            if (board[rookEncodingK] == 1 && board[colourEncodingK] == 0 && board[firstMoveK] == 1 && board[bishopEncodingK] == 0) {
                bool anythingBetween = false;
                for (int betweenSpace=(boardSize-2); betweenSpace>xPos; betweenSpace--) {
                    if (anyPieceHere(board, boardSize, encodingSize, betweenSpace, yPos)) {
                        anythingBetween = true;
                    }
                }
                if (!anythingBetween) {
                    if (!isSquareInCheck(board, boardSize, encodingSize, xPos, yPos) &&
                        !isSquareInCheck(board, boardSize, encodingSize, xPos + 1, yPos) &&
                        !isSquareInCheck(board, boardSize, encodingSize, xPos + 2, yPos)) {
                        int move = yPos * boardSize + xPos + 2;
                        moveLayer[moveFrom + move] = 1;
                    }
                }
            }
        }
    }
    
}

template<typename scalar_t> __global__ void getMoves_cu(
    scalar_t* __restrict__ board,
    scalar_t* __restrict__ moveLayer,
    bool* __restrict__ nothingAllowed,
    bool* __restrict__ everythingAllowed,
    bool* __restrict__ validBlockingSquares,
    const int boardSize,
    const int encodingSize
) {
    getMoves_device_cu(board, moveLayer, nothingAllowed, everythingAllowed, validBlockingSquares, boardSize, encodingSize);
}

template<typename scalar_t> __global__ void getMovesWithSkip_cu(
    scalar_t* __restrict__ board,
    bool* __restrict__ notSkip,
    scalar_t* __restrict__ moveLayer,
    bool* __restrict__ nothingAllowed,
    bool* __restrict__ everythingAllowed,
    bool* __restrict__ validBlockingSquares,
    const int boardSize,
    const int encodingSize
) {
    int thisBatch = blockIdx.x;
    if (notSkip[thisBatch]) {
        getMoves_device_cu(board, moveLayer, nothingAllowed, everythingAllowed, validBlockingSquares, boardSize, encodingSize);
    }
}

template<typename scalar_t> __global__ void getSquaresInCheck_cu(scalar_t* __restrict__ board, scalar_t* __restrict__ checkList, const int boardSize, const int encodingSize) {
    // For debugging isSquareInCheck().
    int thisSquare = blockIdx.x * blockDim.x + threadIdx.x;
    int xPos = threadIdx.x % boardSize;
    int yPos = threadIdx.x / boardSize;
    if (isSquareInCheck(board, boardSize, encodingSize, xPos, yPos)) {
        checkList[thisSquare] = 1;
    }
}

template<typename scalar_t> __global__ void getBlockSquares_cu(
    scalar_t* __restrict__ board,
    scalar_t* __restrict__ blockList,
    bool* __restrict__ nothingAllowed,
    bool* __restrict__ everythingAllowed,
    bool* __restrict__ validBlockingSquares,
    const int boardSize,
    const int encodingSize
) {
    // For debugging getBlockSquares().
    int thisSquare = (blockIdx.x * blockDim.x + threadIdx.x) * boardSize * boardSize;
    int xPos = threadIdx.x % boardSize;
    int yPos = threadIdx.x / boardSize;
    bool* notAllowedPos = &nothingAllowed[thisSquare];
    bool* allAllowedPos = &everythingAllowed[thisSquare];
    bool* validPos = &validBlockingSquares[thisSquare];
    bool* allowed = getBlockSquares(board, boardSize, encodingSize, xPos, yPos, notAllowedPos, allAllowedPos, validPos);
    for (int row=0; row<boardSize; row++) {
        for (int col=0; col<boardSize; col++) {
            int pos = row*boardSize + col;
            if (allowed[pos]) {
                blockList[thisSquare + pos] = 1;
            }
        }
    }
}

template<typename scalar_t> __global__ void getStalemates_cu_(scalar_t* __restrict__ board, bool* __restrict__ gameOverList, bool* __restrict__ stalemateList, const int boardSize, const int encodingSize) {
    int thisBatch = blockIdx.x;
    if (gameOverList[thisBatch]) {
        int thisBoard = blockIdx.x * blockDim.x * encodingSize;
        // Find the King of my colour.
        bool kingFound = false;
        int kingPos = -1;
        for (int row=0; row<boardSize; row++) {
            for (int col=0; col<boardSize; col++) {
                int pos = row*boardSize + col;
                int kingEncodingHere = thisBoard + pos + 4 * blockDim.x;
                int colourOfPiece = thisBoard + pos + 5 * blockDim.x;
                if (board[kingEncodingHere] == 1 && board[colourOfPiece] == 0) {
                    kingFound = true;
                    kingPos = pos;
                    break;
                }
            }
            if (kingFound) {
                break;
            }
        }
        
        if (kingFound) {        
            int kingXPos = kingPos % boardSize;
            int kingYPos = kingPos / boardSize;
            if (!isSquareInCheck(board, boardSize, encodingSize, kingXPos, kingYPos)) {
                // No moves, but not check.
                stalemateList[thisBatch] = 1;
            }
        }
    }
}

template<typename scalar_t> __global__ void selectRandomMoves_cu(scalar_t* __restrict__ moves, scalar_t* __restrict__ randomMove, scalar_t* __restrict__ selectedMoves, const int boardSize) {
    int thisBatch = blockIdx.x;
    int moveN = 0;
    int desiredMove = randomMove[thisBatch];
    int numMoves = boardSize * boardSize * boardSize * boardSize;
    for (int moveNum=0; moveNum<numMoves; moveNum++) {
        if (moves[(thisBatch * numMoves) + moveNum] == 1) {
            if (moveN == desiredMove) {
                int fromSpace = moveNum / (boardSize * boardSize);
                int toSpace = moveNum % (boardSize * boardSize);
                selectedMoves[thisBatch*4] = fromSpace / boardSize;
                selectedMoves[thisBatch*4 + 1] = fromSpace % boardSize;
                selectedMoves[thisBatch*4 + 2] = toSpace / boardSize;
                selectedMoves[thisBatch*4 + 3] = toSpace % boardSize;
                break;
            }
            else{
                moveN++;
            }
        }
    }
}

template<typename scalar_t> __global__ void enactMoves_cu(scalar_t* __restrict__ board, scalar_t* __restrict__ selectedMoves, scalar_t* __restrict__ pawnPromotions, scalar_t* __restrict__ moveCount, const int boardSize, const int encodingSize) {
    int thisBatch = blockIdx.x;
    int thisBoard = thisBatch * encodingSize * boardSize * boardSize;
    int thisMove = thisBatch * 4;
    int thisPromotion = thisBatch * 4;
    int fromPos = selectedMoves[thisMove] * boardSize + selectedMoves[thisMove + 1];
    int toPos = selectedMoves[thisMove + 2] * boardSize + selectedMoves[thisMove + 3];
    int firstMove = board[thisBoard + 6 * boardSize * boardSize + fromPos];
    bool resetMoves = false;

    for (int square=0; square<boardSize*boardSize; square++) {
        board[thisBoard + 7 * boardSize * boardSize + square] = 0;
    }

    // Pawn move.
    if (board[thisBoard + 0 * boardSize * boardSize + fromPos] == 1) {
        // Detect en Passant.
        if (board[thisBoard + 0 * boardSize * boardSize + toPos] == 0 &&
            board[thisBoard + 1 * boardSize * boardSize + toPos] == 0 &&
            board[thisBoard + 2 * boardSize * boardSize + toPos] == 0 &&
            board[thisBoard + 3 * boardSize * boardSize + toPos] == 0 &&
            board[thisBoard + 4 * boardSize * boardSize + toPos] == 0 &&
            board[thisBoard + 5 * boardSize * boardSize + toPos] == 0 &&
            board[thisBoard + 6 * boardSize * boardSize + toPos] == 0 &&
            board[thisBoard + 7 * boardSize * boardSize + toPos] == 0 &&
            (selectedMoves[thisMove + 3] - selectedMoves[thisMove + 1] == 1 ||
            selectedMoves[thisMove + 1] - selectedMoves[thisMove + 3] == 1)) {
            int epPos = (selectedMoves[thisMove + 2] - 1) * boardSize + selectedMoves[thisMove + 3];
            for (int encoding=0; encoding<encodingSize; encoding++) {
                board[thisBoard + encoding * boardSize * boardSize + epPos] = 0;
            }
        }
        
        // Detect pawn promotion.
        if (selectedMoves[thisMove + 2] == boardSize - 1) {
            board[thisBoard + 0 * boardSize * boardSize + fromPos] = 0; // Lose your pawnhood
            if (pawnPromotions[thisPromotion] == 1) {
                board[thisBoard + 1 * boardSize * boardSize + fromPos] = 1; // Gain knighthood
            }
            else if (pawnPromotions[thisPromotion + 1] == 1) {
                board[thisBoard + 2 * boardSize * boardSize + fromPos] = 1; // Gain bishophood
            }
            else if (pawnPromotions[thisPromotion + 2] == 1) {
                board[thisBoard + 3 * boardSize * boardSize + fromPos] = 1; // Gain rookhood
            }
            else if (pawnPromotions[thisPromotion + 3] == 1) {
                board[thisBoard + 2 * boardSize * boardSize + fromPos] = 1; // Gain queenhood
                board[thisBoard + 3 * boardSize * boardSize + fromPos] = 1;
            }
        }
        
        moveCount[thisBatch] = 0;
        resetMoves = true;
    }
    
    // King move.
    if (board[thisBoard + 4 * boardSize * boardSize + fromPos] == 1) {
        // Detect queen-side castling.
        if (selectedMoves[thisMove + 1] - selectedMoves[thisMove + 3] == 2) {
            int qRookFrom = selectedMoves[thisMove] * boardSize;
            int qRookTo = selectedMoves[thisMove] * boardSize + (selectedMoves[thisMove + 1] - 1);
            board[thisBoard + 3 * boardSize * boardSize + qRookTo] = 1;
            board[thisBoard + 7 * boardSize * boardSize + qRookTo] = 1;
            board[thisBoard + 3 * boardSize * boardSize + qRookFrom] = 0;
            board[thisBoard + 6 * boardSize * boardSize + qRookFrom] = 0;
            board[thisBoard + 7 * boardSize * boardSize + qRookFrom] = 0;
        }
        
        // Detect king-side castling.
        if (selectedMoves[thisMove + 3] - selectedMoves[thisMove + 1] == 2) {
            int kRookFrom = selectedMoves[thisMove] * boardSize + (boardSize - 1);
            int kRookTo = selectedMoves[thisMove] * boardSize + (selectedMoves[thisMove + 1] + 1);
            board[thisBoard + 3 * boardSize * boardSize + kRookTo] = 1;
            board[thisBoard + 7 * boardSize * boardSize + kRookTo] = 1;
            board[thisBoard + 3 * boardSize * boardSize + kRookFrom] = 0;
            board[thisBoard + 6 * boardSize * boardSize + kRookFrom] = 0;
            board[thisBoard + 7 * boardSize * boardSize + kRookFrom] = 0;
        }
    }

    for (int encoding=0; encoding<encodingSize; encoding++) {
        int encodingPos = encoding * boardSize * boardSize;
        if (board[thisBoard + encodingPos + toPos] == 1) {
            moveCount[thisBatch] = 0;
            resetMoves = true;
        }
        if (encoding == 6) {
            board[thisBoard + encodingPos + toPos] = 0;
        }
        else{
            board[thisBoard + encodingPos + toPos] = board[thisBoard + encodingPos + fromPos];
        }
        board[thisBoard + encodingPos + fromPos] = 0;
    }

    board[thisBoard + 7 * boardSize * boardSize + toPos] = firstMove;
    if (!resetMoves) {
        moveCount[thisBatch] = moveCount[thisBatch] + 1;
    }
}

template<typename scalar_t> __global__ void getWherePromotion_cu(scalar_t* __restrict__ all_boards, scalar_t* __restrict__ move_per_board, bool* __restrict__ isPromotion, const int boardSize, const int encodingSize) {
    int thisBatch = blockIdx.x;
    int thisBoard = thisBatch * encodingSize * boardSize * boardSize;
    int thisMove = thisBatch * 4;  // 4 is the move length.
    int fromY = move_per_board[thisMove];
    int fromX = move_per_board[thisMove + 1];
    int toY = move_per_board[thisMove + 2];
    int fromPos = fromY * boardSize + fromX;
    if (all_boards[thisBoard + 0 * boardSize * boardSize + fromPos] == 1 &&  // Check for a pawn of our colour
        all_boards[thisBoard + 1 * boardSize * boardSize + fromPos] == 0 &&
        all_boards[thisBoard + 2 * boardSize * boardSize + fromPos] == 0 &&
        all_boards[thisBoard + 3 * boardSize * boardSize + fromPos] == 0 &&
        all_boards[thisBoard + 4 * boardSize * boardSize + fromPos] == 0 &&
        all_boards[thisBoard + 5 * boardSize * boardSize + fromPos] == 0 &&
        toY == 7) {  // Check that it's going to the end of the board.
        isPromotion[thisBatch] = 1;
    }
}

template<typename scalar_t> __global__ void expandPromotions_cu(scalar_t* __restrict__ promotion_per_board, scalar_t* __restrict__ newPromotions, bool* __restrict__ promoteMask) {
    int thisBatch = blockIdx.x;
    int thisArea = thisBatch * 4;
    int newIndex = 0;
    for (int i=0; i<thisBatch; i++) {
        newIndex += 1;
        if (promoteMask[i] == 1) {
            newIndex += 3;
        }
    }
    int newArea = newIndex * 4;
    if (promoteMask[thisBatch] == 1) {
        newPromotions[newArea] = 1;
        newPromotions[newArea + 1] = 0;
        newPromotions[newArea + 2] = 0;
        newPromotions[newArea + 3] = 0;
        
        newPromotions[newArea + 4] = 0;
        newPromotions[newArea + 5] = 1;
        newPromotions[newArea + 6] = 0;
        newPromotions[newArea + 7] = 0;
        
        newPromotions[newArea + 8] = 0;
        newPromotions[newArea + 9] = 0;
        newPromotions[newArea + 10] = 1;
        newPromotions[newArea + 11] = 0;
        
        newPromotions[newArea + 12] = 0;
        newPromotions[newArea + 13] = 0;
        newPromotions[newArea + 14] = 0;
        newPromotions[newArea + 15] = 1;
    }
    else {
        for (int i=0; i<4; i++) {
            newPromotions[newArea + i] = promotion_per_board[thisArea + i];
        }
    }
}

template<typename scalar_t> __global__ void expandBoards_cu(scalar_t* __restrict__ all_boards, scalar_t* __restrict__ newBoards, bool* __restrict__ promoteMask, const int boardSize, const int encodingSize) {
    int thisBatch = blockIdx.x;
    int thisArea = thisBatch * encodingSize * boardSize * boardSize;
    int newIndex = 0;
    for (int i=0; i<thisBatch; i++) {
        newIndex += 1;
        if (promoteMask[i] == 1) {
            newIndex += 3;
        }
    }
    int newArea = newIndex * encodingSize * boardSize * boardSize;
    if (promoteMask[thisBatch] == 1) {
        for (int j=0; j<4; j++) {
            for (int i=0; i<encodingSize * boardSize * boardSize; i++) {
                newBoards[newArea + i + j * encodingSize * boardSize * boardSize] = all_boards[thisArea + i];
            }
        }
    }
    else {
        for (int i=0; i<encodingSize * boardSize * boardSize; i++) {
            newBoards[newArea + i] = all_boards[thisArea + i];
        }
    }
}

template<typename scalar_t> __global__ void expandValidProbs_cu(float* __restrict__ valid_probs, float* __restrict__ newValidProbs, bool* __restrict__ promoteMask) {
    int thisBatch = blockIdx.x;
    int newIndex = 0;
    for (int i=0; i<thisBatch; i++) {
        newIndex += 1;
        if (promoteMask[i] == 1) {
            newIndex += 3;
        }
    }
    if (promoteMask[thisBatch] == 1) {
        float previous_valid_prob = valid_probs[thisBatch];
        newValidProbs[newIndex] = previous_valid_prob / 4;
        newValidProbs[newIndex + 1] = previous_valid_prob / 4;
        newValidProbs[newIndex + 2] = previous_valid_prob / 4;
        newValidProbs[newIndex + 3] = previous_valid_prob / 4;
    }
    else {
        newValidProbs[newIndex] = valid_probs[thisBatch];
    }
}

void getMoves(torch::Tensor board, torch::Tensor moveLayer, const int batch_size, const int boardSize, const int encodingSize) {
    size_t threadsPerBlock = boardSize * boardSize; // Could go up to 256. Number of squares (64).
    size_t numberOfBlocks = batch_size; // Could go up to 2176. Batch size (256).
    bool *nothingAllowed;
    bool *everythingAllowed;
    bool *validBlockingSquares;
    cudaMallocManaged(&nothingAllowed, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    cudaMallocManaged(&everythingAllowed, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    cudaMallocManaged(&validBlockingSquares, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    
    // From here, I will need to test for check and then get moves.

    AT_DISPATCH_INTEGRAL_TYPES(board.type(), "getMoves_cu_", ([&] {
        getMoves_cu<scalar_t><<<numberOfBlocks, threadsPerBlock>>>(
            board.data<scalar_t>(),
            moveLayer.data<scalar_t>(),
            nothingAllowed,
            everythingAllowed,
            validBlockingSquares,
            boardSize, // Currently assumes number of rows = number of columns
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
	cudaFree(nothingAllowed);
	cudaFree(everythingAllowed);
	cudaFree(validBlockingSquares);
}

void getMovesWithSkip(torch::Tensor board, torch::Tensor moveLayer, const int batch_size, const int boardSize, const int encodingSize, torch::Tensor notSkip) {
    size_t threadsPerBlock = boardSize * boardSize; // Could go up to 256. Number of squares (64).
    size_t numberOfBlocks = batch_size; // Could go up to 2176. Batch size (256).
    bool *nothingAllowed;
    bool *everythingAllowed;
    bool *validBlockingSquares;
    cudaMallocManaged(&nothingAllowed, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    cudaMallocManaged(&everythingAllowed, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    cudaMallocManaged(&validBlockingSquares, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    
    // From here, I will need to test for check and then get moves.

    AT_DISPATCH_INTEGRAL_TYPES(board.type(), "getMovesWithSkip_cu_", ([&] {
        getMovesWithSkip_cu<scalar_t><<<numberOfBlocks, threadsPerBlock>>>(
            board.data<scalar_t>(),
            notSkip.data<bool>(),
            moveLayer.data<scalar_t>(),
            nothingAllowed,
            everythingAllowed,
            validBlockingSquares,
            boardSize, // Currently assumes number of rows = number of columns
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
	cudaFree(nothingAllowed);
	cudaFree(everythingAllowed);
	cudaFree(validBlockingSquares);
}

void getStalemates(torch::Tensor board, torch::Tensor gameOverList, torch::Tensor stalemateList, const int batch_size, const int boardSize, const int encodingSize) {
    size_t threadsPerBlock = boardSize * boardSize;
    size_t numberOfBlocks = batch_size;

    AT_DISPATCH_INTEGRAL_TYPES(board.type(), "getStalemates_cu_", ([&] {
        getStalemates_cu_<scalar_t><<<numberOfBlocks, threadsPerBlock>>>(
            board.data<scalar_t>(),
            gameOverList.data<bool>(),
            stalemateList.data<bool>(),
            boardSize,
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
}

void getSquaresInCheck(torch::Tensor board, torch::Tensor checkList, const int batch_size, const int boardSize, const int encodingSize) {
    // For debugging isSquareInCheck().
    size_t threadsPerBlock = boardSize * boardSize;
    size_t numberOfBlocks = batch_size;

    AT_DISPATCH_INTEGRAL_TYPES(board.type(), "getSquaresInCheck_cu_", ([&] {
        // This kernel needs to be the same as the one above so that getSquaresInCheck is run under the same conditions as in getMoves.
        getSquaresInCheck_cu<scalar_t><<<numberOfBlocks, threadsPerBlock>>>(
            board.data<scalar_t>(),
            checkList.data<scalar_t>(),
            boardSize,
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
}

void getValidBlockingSquares(torch::Tensor board, torch::Tensor blockList, const int batch_size, const int boardSize, const int encodingSize) {
    // For debugging getBlockSquares().
    size_t threadsPerBlock = boardSize * boardSize;
    size_t numberOfBlocks = batch_size;
    bool *nothingAllowed;
    bool *everythingAllowed;
    bool *validBlockingSquares;
    cudaMallocManaged(&nothingAllowed, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    cudaMallocManaged(&everythingAllowed, batch_size * (boardSize * boardSize) * (boardSize * boardSize));
    cudaMallocManaged(&validBlockingSquares, batch_size * (boardSize * boardSize) * (boardSize * boardSize));

    AT_DISPATCH_INTEGRAL_TYPES(board.type(), "getBlockSquares_cu", ([&] {
        // This kernel needs to be the same as the one above so that getBlockSquares is run under the same conditions as in getMoves.
        getBlockSquares_cu<scalar_t><<<numberOfBlocks, threadsPerBlock>>>(
            board.data<scalar_t>(),
            blockList.data<scalar_t>(),
            nothingAllowed,
            everythingAllowed,
            validBlockingSquares,
            boardSize,
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
	cudaFree(nothingAllowed);
	cudaFree(everythingAllowed);
	cudaFree(validBlockingSquares);
}

void selectRandomMoves(torch::Tensor moves, torch::Tensor randomMove, torch::Tensor selectedMoves, const int batch_size, const int boardSize) {
    size_t numberOfBlocks = batch_size;
    
    AT_DISPATCH_INTEGRAL_TYPES(randomMove.type(), "selectRandomMoves_cu", ([&] {
        selectRandomMoves_cu<scalar_t><<<numberOfBlocks, 1>>>(
            moves.data<scalar_t>(),
            randomMove.data<scalar_t>(),
            selectedMoves.data<scalar_t>(),
            boardSize
		);
    }));

	cudaDeviceSynchronize();
}

void enactMoves(torch::Tensor board, torch::Tensor selectedMoves, torch::Tensor pawnPromotions, torch::Tensor moveCount, const int batch_size, const int boardSize, const int encodingSize) {
    size_t numberOfBlocks = batch_size;
    
    AT_DISPATCH_INTEGRAL_TYPES(board.type(), "enactMoves_cu", ([&] {
        enactMoves_cu<scalar_t><<<numberOfBlocks, 1>>>(
            board.data<scalar_t>(),
            selectedMoves.data<scalar_t>(),
            pawnPromotions.data<scalar_t>(),
            moveCount.data<scalar_t>(),
            boardSize,
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
}

void getWherePromotion(torch::Tensor all_boards, torch::Tensor move_per_board, torch::Tensor isPromotion, const int batch_size, const int boardSize, const int encodingSize) {
    size_t numberOfBlocks = batch_size;

    AT_DISPATCH_INTEGRAL_TYPES(move_per_board.type(), "getWherePromotion_cu", ([&] {
        getWherePromotion_cu<scalar_t><<<numberOfBlocks, 1>>>(
            all_boards.data<scalar_t>(),
            move_per_board.data<scalar_t>(),
            isPromotion.data<bool>(),
            boardSize,
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
}

void expandPromotions(torch::Tensor promotion_per_board, torch::Tensor newPromotions, torch::Tensor promoteMask, const int batch_size) {
    size_t numberOfBlocks = batch_size;

    AT_DISPATCH_INTEGRAL_TYPES(promotion_per_board.type(), "expandPromotions_cu", ([&] {
        expandPromotions_cu<scalar_t><<<numberOfBlocks, 1>>>(
            promotion_per_board.data<scalar_t>(),
            newPromotions.data<scalar_t>(),
            promoteMask.data<bool>()
		);
    }));

	cudaDeviceSynchronize();
}

void expandBoards(torch::Tensor all_boards, torch::Tensor newBoards, torch::Tensor promoteMask, const int batch_size, const int boardSize, const int encodingSize) {
    size_t numberOfBlocks = batch_size;

    AT_DISPATCH_INTEGRAL_TYPES(all_boards.type(), "expandBoards_cu", ([&] {
        expandBoards_cu<scalar_t><<<numberOfBlocks, 1>>>(
            all_boards.data<scalar_t>(),
            newBoards.data<scalar_t>(),
            promoteMask.data<bool>(),
            boardSize,
            encodingSize
		);
    }));

	cudaDeviceSynchronize();
}

void expandValidProbs(torch::Tensor valid_probs, torch::Tensor newValidProbs, torch::Tensor promoteMask, const int batch_size) {
    size_t numberOfBlocks = batch_size;

    AT_DISPATCH_ALL_TYPES(valid_probs.type(), "expandValidProbs_cu", ([&] {
        expandValidProbs_cu<float><<<numberOfBlocks, 1>>>(
            valid_probs.data<float>(),
            newValidProbs.data<float>(),
            promoteMask.data<bool>()
		);
    }));

	cudaDeviceSynchronize();
}
