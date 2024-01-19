from chessboard import display
import tkinter as tk

fen_list = [
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',
    'rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1',
    # Add more FEN strings as needed
]

def update_board(index):
    display.update(fen_list[index], game_board)
    root.after(1000, update_board, (index + 1) % len(fen_list))

# Create a Tkinter root window but hide it
root = tk.Tk()
root.withdraw()

# Create the chessboard
game_board = display.start()

# Initial board update
update_board(0)

# board flip interface
if not game_board.flipped:
    display.flip(game_board)

root.mainloop()