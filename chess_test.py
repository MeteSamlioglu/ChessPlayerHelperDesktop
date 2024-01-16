import chess
import copy



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