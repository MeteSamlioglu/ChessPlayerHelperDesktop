import chess
import copy

def get_total_piece_count(chess_board):
    piece_count = 0

    # Iterate over the piece_map to count pieces
    for square, piece in chess_board.piece_map().items():
        if piece is not None:
            piece_count += 1

    return piece_count
    

board = chess.Board()

uci_move = "b8c6"

copied_board = copy.deepcopy(board)

move = chess.Move.from_uci(uci_move)

board.push(move)

print(board)

print("----------------------")

print(copied_board)

square_index = 63

# Convert numerical index to algebraic notation
square_algebraic = chess.square_name(square_index)
print(square_algebraic)


count = get_total_piece_count(board)
print(count)



integer_square = 63
algebraic_square = chess.square_name(integer_square)
print(algebraic_square)  # Output: h1
