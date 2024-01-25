import chess
from stockfish import Stockfish

board = chess.Board()
print(board)

piece_at = board.piece_at(30)
print(piece_at)

fen_position = "r1bqkbnr/ppp1pppp/2n5/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3"
board = chess.Board(fen_position)

print(board)

move = chess.Move.from_uci("f3e5")  # Example move: Nf3
print("------------------------------------")

board.push(move)
print(board)

print("---------------------------------")
board.pop()
print(board)
# if board.is_legal(move):
#     if board.is_castling(move):
#         print("Castling")
#         if board.is_kingside_castling(move):
#             print("Kingside castling")
#         if board.is_queenside_castling(move):
#             print("Queenside castling")
#     else:
#         print("It is not a castling")
    
#     board.push(move)
#     print("\nPosition after making the legal move:")
#     print(board)

# else:
#     print("\nThe move is not legal.")


# queenside_castling = "r1bqk2r/1ppp1ppp/p1n1pn2/8/1bB1P3/5N2/PPPP1PPP/RNBQK2R w - - 0 2"

# queenside_castling_board = chess.Board(queenside_castling)

# print("Board")
# print(queenside_castling_board)

# move1 = chess.Move.from_uci("e1a1")

# if queenside_castling_board.has_castling_rights(chess.WHITE):
#     print("White has castling rights")
# else:
#     print("White does not have a castling rights")

# if queenside_castling_board.is_castling(move1):
#     print("Castling")
#     if queenside_castling_board.is_kingside_castling(move1):
#         print("Kingside castling")
#     if queenside_castling_board.is_queenside_castling(move1):
#         print("Queenside castling")
    
#     queenside_castling_board.push(move1)
#     print("\nPosition after making the legal move:")
#     print(queenside_castling_board)

# else:
#     print("It is not a castling")
    

# stockfish = Stockfish(r'D:\\chesscog\\stockfish\\stockfish-windows-x86-64-avx2')

# starting_fen_position = "rnbqkbnr/1p2pppp/p7/2pp4/3P1BQ1/4P3/PPP2PPP/RN2KBNR b KQkq - 1 4"

# # Set the FEN position to Stockfish
# stockfish.set_fen_position(starting_fen_position)

# # Apply the move to the board
# board = chess.Board(starting_fen_position)
# print("Starting Fen Position")
# print(board)

# best_move = stockfish.get_best_move()
# print("Best move")
# print(best_move)
# print(type(best_move))
# evaluation = stockfish.get_evaluation()
# print(f"Evaluation after your move: {evaluation}")

# my_move = chess.Move.from_uci("e2e4")

# board.push(my_move)

# print("After f3e5 ")
# print(board)


# Set the updated position to Stockfish
# stockfish.set_fen_position(board.fen())

# Get the evaluation after your move
# evaluation = stockfish.get_evaluation()
# print(f"Evaluation after your move: {evaluation}")

# # Interpret the evaluation
# if evaluation["value"] > 0:
#     print("Brilliant Move!")
# elif -1 <= evaluation["value"] <= 1:
#     print("Correct Move")
# elif evaluation["value"] < -1:
#     print("Blunder")