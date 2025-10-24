#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <torch/extension.h>
#include <stdint.h>
// uncomment to disable assert()
// #define NDEBUG
 
// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// CUDA forward declarations
void getMoves(torch::Tensor board, torch::Tensor moveLayer, const int batch_size, const int boardSize, const int encodingSize);
void getMovesWithSkip(torch::Tensor board, torch::Tensor moveLayer, const int batch_size, const int boardSize, const int encodingSize, torch::Tensor notSkip);
void getSquaresInCheck(torch::Tensor board, torch::Tensor checkList, const int batch_size, const int boardSize, const int encodingSize);
void getValidBlockingSquares(torch::Tensor board, torch::Tensor blockList, const int batch_size, const int boardSize, const int encodingSize);
void getStalemates(torch::Tensor board, torch::Tensor gameOverList, torch::Tensor stalemateList, const int batch_size, const int boardSize, const int encodingSize);
void selectRandomMoves(torch::Tensor flatMoves, torch::Tensor randomMove, torch::Tensor selectedMoves, const int batch_size, const int boardSize);
void enactMoves(torch::Tensor board, torch::Tensor selectedMoves, torch::Tensor pawnPromotions, torch::Tensor moveCount, const int batch_size, const int boardSize, const int encodingSize);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
const int num_rows = 8;
const int num_cols = 8;
const int pos_encoding = 6;
const int bits_per_piece = pos_encoding + 2;

//// Pieces - For human readability and testing only.

class Piece {
    private:
        char kind;
        bool colour;
        int posX;
        int posY;
        bool notMovedYet;
        bool justUsedFirstMove;
    public:
        Piece(char kind, bool colour, int posX, int posY) {
            this->kind = kind;
            this->colour = colour;
            this->posX = posX;
            this->posY = posY;
            this->notMovedYet = true;
            this->justUsedFirstMove = false;
        }
        Piece(char kind, bool colour, int posX, int posY, bool notMovedYet, bool justUsedFirstMove) {
            // To be used for unit testing.
            this->kind = kind;
            this->colour = colour;
            this->posX = posX;
            this->posY = posY;
            this->notMovedYet = notMovedYet;
            this->justUsedFirstMove = justUsedFirstMove;
        }
        char getKind() {
            return kind;
        }
        bool getColour() {
            return colour;
        }
        int getPosX() {
            return posX;
        }
        int getPosY() {
            return posY;
        }
        void enactFirstMove() {
            assertm(this->notMovedYet, "Must be called on first move only.");
            this->notMovedYet = false;
            this->justUsedFirstMove = true;
        }
        void enactMove() {
            assertm(!this->notMovedYet, "Cannot be called on first move.");
            this->justUsedFirstMove = false;
        }
        bool isFirstMove() {
            return notMovedYet;
        }
        bool hasJustUsedFirstMove() {
            return justUsedFirstMove;
        }
};

class Square {
    private:
        Piece piece = Piece('e', true, 0, 0);
        int batch_size;
    public:
        Square(int batch_size) {
            this->batch_size = batch_size;
        }
        int get_batch_size() {
            return this->batch_size;
        }
        void setPiece(Piece piece) {
            this->piece = piece;
        }
        Piece getPiece() {
            return piece;
        }
};


class BatchedBoard {
    private:
        std::vector<std::vector<std::vector<Square>>> squares;
        std::vector<std::unordered_map<std::string, int>> threefoldRepetitions;
        int batch_size;
        void setupBoard() {
            // For each square in each batch, setup the usual starting pieces here.
        }
    public:
        BatchedBoard(bool setup_starting_pieces, int batch_size, int configuration) {
            this->batch_size = batch_size;
            // Hard-code 8 here.
            for (int batch=0; batch<batch_size; batch++) {
                std::vector<std::vector<Square>> thisBoard;
                for (int row=0; row<num_rows; row++) {
                    std::vector<Square> wholeCol;
                    for (int col=0; col<num_cols; col++) {
                        wholeCol.push_back(Square(batch_size));
                    }
                    thisBoard.push_back(wholeCol);
                }
                squares.push_back(thisBoard);
                
                std::unordered_map<std::string, int> repetitions;
                threefoldRepetitions.push_back(repetitions);
            }
            if (setup_starting_pieces && configuration == 0) {
                // Full chess
                for (int batch=0; batch<batch_size; batch++) {
                    squares[batch][0][0].setPiece(Piece('r', true, 0, 0));
                    squares[batch][0][1].setPiece(Piece('n', true, 0, 1));
                    squares[batch][0][2].setPiece(Piece('b', true, 0, 2));
                    squares[batch][0][3].setPiece(Piece('q', true, 0, 3));
                    squares[batch][0][4].setPiece(Piece('k', true, 0, 4));
                    squares[batch][0][5].setPiece(Piece('b', true, 0, 5));
                    squares[batch][0][6].setPiece(Piece('n', true, 0, 6));
                    squares[batch][0][7].setPiece(Piece('r', true, 0, 7));
                    squares[batch][1][0].setPiece(Piece('p', true, 1, 0));
                    squares[batch][1][1].setPiece(Piece('p', true, 1, 1));
                    squares[batch][1][2].setPiece(Piece('p', true, 1, 2));
                    squares[batch][1][3].setPiece(Piece('p', true, 1, 3));
                    squares[batch][1][4].setPiece(Piece('p', true, 1, 4));
                    squares[batch][1][5].setPiece(Piece('p', true, 1, 5));
                    squares[batch][1][6].setPiece(Piece('p', true, 1, 6));
                    squares[batch][1][7].setPiece(Piece('p', true, 1, 7));
                    
                    squares[batch][7][0].setPiece(Piece('r', false, 7, 0));
                    squares[batch][7][1].setPiece(Piece('n', false, 7, 1));
                    squares[batch][7][2].setPiece(Piece('b', false, 7, 2));
                    squares[batch][7][3].setPiece(Piece('q', false, 7, 3));
                    squares[batch][7][4].setPiece(Piece('k', false, 7, 4));
                    squares[batch][7][5].setPiece(Piece('b', false, 7, 5));
                    squares[batch][7][6].setPiece(Piece('n', false, 7, 6));
                    squares[batch][7][7].setPiece(Piece('r', false, 7, 7));
                    squares[batch][6][0].setPiece(Piece('p', false, 6, 0));
                    squares[batch][6][1].setPiece(Piece('p', false, 6, 1));
                    squares[batch][6][2].setPiece(Piece('p', false, 6, 2));
                    squares[batch][6][3].setPiece(Piece('p', false, 6, 3));
                    squares[batch][6][4].setPiece(Piece('p', false, 6, 4));
                    squares[batch][6][5].setPiece(Piece('p', false, 6, 5));
                    squares[batch][6][6].setPiece(Piece('p', false, 6, 6));
                    squares[batch][6][7].setPiece(Piece('p', false, 6, 7));
                }
            }
            if (setup_starting_pieces && configuration == 1) {
                // Highly simplified chess
                for (int batch=0; batch<batch_size; batch++) {
                    squares[batch][0][0].setPiece(Piece('k', true, 0, 0));
                    squares[batch][0][1].setPiece(Piece('r', true, 0, 1));
                    squares[batch][1][0].setPiece(Piece('q', true, 1, 0));
                    
                    squares[batch][7][7].setPiece(Piece('k', false, 7, 7));
                    squares[batch][7][6].setPiece(Piece('r', false, 7, 6));
                    squares[batch][6][7].setPiece(Piece('q', false, 6, 7));
                }
            }
        }

        std::vector<std::vector<std::vector<Square>>> get_squares() {
            return squares;
        }

        torch::Tensor to_tensor() {
            // Pawn -> 1, Knight -> 2, Bishop -> 4, Rook -> 8, King -> 16
            // Queen -> 4 + 8 = 12
            // My pieces -> +0, Opponent's pieces -> +32
            std::vector<int64_t> tensor_size = {this->batch_size, bits_per_piece, num_rows, num_cols};
            torch::Tensor board = torch::zeros(tensor_size).to(torch::kInt8);
            for (int batch=0; batch<this->batch_size; batch++) {
                // board[batch][3][3][3] = 1; // Debugging piece.
                // board[batch][2][3][3] = 1; // Debugging piece.
                for (int row=0; row<num_rows; row++) {
                    for (int col=0; col<num_cols; col++) {
                        // if (typeid(squares) == typeid(int)) if for each piece.
                        // Possibly move to CUDA?
                        if (squares[batch][row][col].getPiece().getKind() == 'p') {
                            board[batch][0][row][col] = 1;
                        }
                        if (squares[batch][row][col].getPiece().getKind() == 'n') {
                            board[batch][1][row][col] = 1;
                        }
                        if (squares[batch][row][col].getPiece().getKind() == 'b') {
                            board[batch][2][row][col] = 1;
                        }
                        if (squares[batch][row][col].getPiece().getKind() == 'r') {
                            board[batch][3][row][col] = 1;
                        }
                        if (squares[batch][row][col].getPiece().getKind() == 'k') {
                            board[batch][4][row][col] = 1;
                        }
                        if (squares[batch][row][col].getPiece().getKind() == 'q') {
                            board[batch][2][row][col] = 1;
                            board[batch][3][row][col] = 1;
                        }
                        if (!squares[batch][row][col].getPiece().getColour()) {
                            board[batch][5][row][col] = 1;
                        }
                        if (squares[batch][row][col].getPiece().isFirstMove()) {
                            if (squares[batch][row][col].getPiece().getKind() != 'e') {
                                board[batch][6][row][col] = 1;
                            }
                        }
                        if (squares[batch][row][col].getPiece().hasJustUsedFirstMove()) {
                            if (squares[batch][row][col].getPiece().getKind() != 'e') {
                                board[batch][7][row][col] = 1;
                            }
                        }
                    }
                }
            }
            return board;
        }

        void setPiece(int batch, Piece piece) {
            squares[batch][piece.getPosX()][piece.getPosY()].setPiece(piece);
        }

        torch::Tensor get_starting_move_count_list() {
            std::vector<int64_t> tensor_size = {batch_size, 1};
            auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
            torch::Tensor moveCount = torch::zeros(tensor_size, options);
            return moveCount;
        }
        
        torch::Tensor check_threefold_repetition(torch::Tensor compact_pos) {
            const int batch_size = compact_pos.sizes()[0];
            std::vector<int64_t> tensor_size = {batch_size, 1};
            auto options = torch::TensorOptions().dtype(torch::kBool);
            torch::Tensor gameOver = torch::zeros(tensor_size, options);

            for (int bNum=0; bNum<this->batch_size; bNum++) {
                int8_t* int8_pos_batch = compact_pos[bNum].contiguous().data_ptr<int8_t>();
                char* char_pos_batch = reinterpret_cast<char*>(int8_pos_batch);
                std::string moveStr(char_pos_batch, (num_rows + 2) * pos_encoding);
                std::unordered_map<std::string, int> thisMap = this->threefoldRepetitions.at(bNum);
                if (thisMap.find(moveStr) == thisMap.end()) {
                    thisMap[moveStr] = 1;
                }
                else{
                    thisMap[moveStr] = thisMap[moveStr] + 1;
                }
                int numSeen = thisMap[moveStr];
                if (numSeen >= 3) {
                    gameOver[bNum] = true;
                }
                this->threefoldRepetitions[bNum] = thisMap;
            }
            
            torch::Device device(torch::kCUDA);
            gameOver = gameOver.to(device);
            return gameOver;
        }
        
        void reset_repetitions(torch::Tensor is_game_over) {
            for (int bNum=0; bNum<this->batch_size; bNum++) {
                if (is_game_over[bNum].item<bool>() == true) {
                    this->threefoldRepetitions[bNum].clear();
                }
            }
        }
        
        void update_batch_size(int new_batch_size) {
            this->batch_size = new_batch_size;
        }
};

torch::Tensor get_moves_for_player(torch::Tensor board) {
    // Let moves be defined as ints between 0 and 4095. Let their definition be 64*(start) + end.
    // Let start or end be defined as 8*row (letter) + col (number).
    const int batch_size = board.sizes()[0];
    std::vector<int64_t> tensor_size = {batch_size, num_rows * num_cols, num_rows * num_cols};
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    torch::Tensor moveLayer = torch::zeros(tensor_size, options);
    getMoves(board, moveLayer, batch_size, num_rows, bits_per_piece);
    return moveLayer;
}

torch::Tensor get_moves_for_player_with_skip(torch::Tensor board, torch::Tensor not_skip) {
    // Let moves be defined as ints between 0 and 4095. Let their definition be 64*(start) + end.
    // Let start or end be defined as 8*row (letter) + col (number).
    const int batch_size = board.sizes()[0];
    std::vector<int64_t> tensor_size = {batch_size, num_rows * num_cols, num_rows * num_cols};
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    torch::Tensor moveLayer = torch::zeros(tensor_size, options);
    getMovesWithSkip(board, moveLayer, batch_size, num_rows, bits_per_piece, not_skip);
    return moveLayer;
}

torch::Tensor get_squares_in_check_for_player(torch::Tensor board) {
    const int batch_size = board.sizes()[0];
    std::vector<int64_t> tensor_size = {batch_size, num_rows * num_cols};
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    torch::Tensor checkList = torch::zeros(tensor_size, options);
    getSquaresInCheck(board, checkList, batch_size, num_rows, bits_per_piece);
    return checkList;
}

torch::Tensor get_valid_blocking_squares_player(torch::Tensor board) {
    const int batch_size = board.sizes()[0];
    std::vector<int64_t> tensor_size = {batch_size, num_rows * num_cols, num_rows * num_cols};
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    torch::Tensor blockList = torch::zeros(tensor_size, options);
    getValidBlockingSquares(board, blockList, batch_size, num_rows, bits_per_piece);
    return blockList;
}

torch::Tensor is_stalemate(torch::Tensor board, torch::Tensor gameOverList) {
    const int batch_size = board.sizes()[0];
    std::vector<int64_t> tensor_size = {batch_size, 1};
    auto options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    torch::Tensor stalemateList = torch::zeros(tensor_size, options);
    getStalemates(board, gameOverList, stalemateList, batch_size, num_rows, bits_per_piece);
    return stalemateList;
}

torch::Tensor get_random_valid_move_per_game(torch::Tensor flatMoves, torch::Tensor randomMove) {
    // Get the randomMove'th element in each batch in terms of (from y, from x, to y, to x).
    const int batch_size = flatMoves.sizes()[0];
    std::vector<int64_t> tensor_size = {batch_size, 4};
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    torch::Tensor selectedMoves = torch::zeros(tensor_size, options);
    selectRandomMoves(flatMoves, randomMove, selectedMoves, batch_size, num_rows);
    return selectedMoves;
}

void enact_moves(torch::Tensor board, torch::Tensor selected_moves, torch::Tensor pawn_promotions, torch::Tensor move_count) {
    const int batch_size = board.sizes()[0];
    enactMoves(board, selected_moves, pawn_promotions, move_count, batch_size, num_rows, bits_per_piece);
}

//// Pybind definitions

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Square>(m, "Square")
        .def("get_batch_size", &Square::get_batch_size)
        .def("get_piece", &Square::getPiece)
        .def(py::init<int>());
    py::class_<BatchedBoard>(m, "BatchedBoard")
        .def("get_squares", &BatchedBoard::get_squares)
        .def("setPiece", &BatchedBoard::setPiece)
        .def("to_tensor", &BatchedBoard::to_tensor)
        .def("check_threefold_repetition", &BatchedBoard::check_threefold_repetition)
        .def("reset_repetitions", &BatchedBoard::reset_repetitions)
        .def("get_starting_move_count_list", &BatchedBoard::get_starting_move_count_list)
        .def("update_batch_size", &BatchedBoard::update_batch_size)
        .def(py::init<bool, int, int>());
    py::class_<Piece>(m, "Piece")
        .def("get_kind", &Piece::getKind)
        .def("get_colour", &Piece::getColour)
        .def("get_posX", &Piece::getPosX)
        .def("get_posY", &Piece::getPosY)
        .def("isFirstMove", &Piece::isFirstMove)
        .def(py::init<char, bool, int, int>())
        .def(py::init<char, bool, int, int, bool, bool>());
    m.def("get_moves_for_player", &get_moves_for_player, "get_moves_for_player");
    m.def("get_moves_for_player_with_skip", &get_moves_for_player_with_skip, "get_moves_for_player_with_skip");
    m.def("get_squares_in_check_for_player", &get_squares_in_check_for_player, "get_squares_in_check_for_player");
    m.def("get_valid_blocking_squares_player", &get_valid_blocking_squares_player, "get_valid_blocking_squares_player");
    m.def("get_random_valid_move_per_game", &get_random_valid_move_per_game, "get_random_valid_move_per_game");
    m.def("enact_moves", &enact_moves, "enact_moves");
    m.def("is_stalemate", &is_stalemate, "is_stalemate");
}
