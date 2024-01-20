import chess
from stockfish import Stockfish

board = chess.Board()
print(board)

piece_at = board.piece_at(30)
print(piece_at)
# stockfish = Stockfish(r'D:\\chesscog\\stockfish\\stockfish-windows-x86-64-avx2')

# starting_fen_position = "rn1qkb1r/pp1b1ppp/4p3/1BP5/8/2P1BN2/P1P2PPP/R2QK2R w KQkq - 2 10"

# # Set the FEN position to Stockfish
# stockfish.set_fen_position(starting_fen_position)



# Make your own move

# # Apply the move to the board
# board = chess.Board(starting_fen_position)
# print("Starting Fen Position")
# print(board)

# best_move = stockfish.get_best_move()
# print("Best move")
# print(best_move)

# my_move = chess.Move.from_uci("f3e5")

# board.push(my_move)
# print("After f3e5 ")
# print(board)


# # Set the updated position to Stockfish
# stockfish.set_fen_position(board.fen())

# # Get the evaluation after your move
# evaluation = stockfish.get_evaluation()
# print(f"Evaluation after your move: {evaluation}")

# # Interpret the evaluation
# if evaluation["value"] > 0:
#     print("Brilliant Move!")
# elif -1 <= evaluation["value"] <= 1:
#     print("Correct Move")
# elif evaluation["value"] < -1:
#     print("Blunder")