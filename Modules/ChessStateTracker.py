import chess
from chess import Status


class StateTracker:
    
    THRESHOLD_VALUE = 4
    
    def __init__(self):
        
        self.States = []
        self.state_counter = 0
        
        self.WHITE_TURN = True
        self.BLACK_TURN = False
        
        if self.WHITE_TURN == True:
            self.white_initialized_board = chess.Board()
        
    def add_state(self, CurrentState):
        
        if self.state_counter == 0:
            if CurrentState !=  self.white_initialized_board:
                print("======== Initial board is not recognized properly ========")
                return False
            else:
               print("======== Initial board is recognized  ========")
               self.States.append(self.white_initialized_board)
               self.state_counter+=1
               return True
        
        previous_state = self.States[self.state_counter - 1]
        
        # differences = list(CurrentState.piece_map().items() - previous_state.piece_map().items())
        
        # print("Differences:")
        # for square, piece in differences:
        #     print(f"At square {chess.square_name(square)}, board1 has {piece} and board2 has {previous_state.piece_at(square)}.")
        # Find differences in white pieces
        print(f'State Number {self.state_counter}')
        if self.WHITE_TURN == True:
            
            white_differences = self.find_difference(previous_state, CurrentState, chess.BLACK)
            
            white_diff_count = len(white_differences)
            
            print("White's Turn")
            print(f'Number diff moves {white_diff_count}')
            print("Differences in White Pieces:")
            for square, current_piece, previous_piece in white_differences:
                print(f"At square {chess.square_name(square)}, current board has {current_piece} and previous board has {previous_piece}.")
            
            self.WHITE_TURN = False
            self.BLACK_TURN = True
            
            self.States.append(CurrentState)
            self.state_counter+=1
            
            print(CurrentState)
            print("---------------------------")
            if white_diff_count > StateTracker.THRESHOLD_VALUE: 
                return False
            
            return True
        
        if self.BLACK_TURN == True:
            black_differences = self.find_difference(previous_state, CurrentState, chess.WHITE)
            black_diff_count = len(black_differences)
            print("Black's Turn")
            print(f'Number diff moves {black_diff_count}')
            print("Differences in Black Pieces:")
            for square, current_piece, previous_piece in black_differences:
                print(f"At square {chess.square_name(square)}, current board has {current_piece} and previous board has {previous_piece}.")
            
            self.WHITE_TURN = True
            self.BLACK_TURN = False   
            
            self.States.append(CurrentState)
            self.state_counter+=1
            
            print(CurrentState)
            print("---------------------------")   
            
            if black_diff_count > StateTracker.THRESHOLD_VALUE: 
                return False
            
            return True

        """ 
            Method takes previous and current state of the board and finds the difference except the given parameter color
            Param:
                PreviousState : The previous state of the game
                CurrentState  : The current state of the game
                Color : Piece type that is going to be ignored when searching for the difference 

            Returns:
                Returns the difference between the current and the previous board for given parameter color
        """
    
    def find_difference(self, PreviousState, CurrentState, color : chess.Color) -> list[tuple]:
        
        differences = [
            (square, PreviousState.piece_at(square), CurrentState.piece_at(square))
            for square in chess.SQUARES
            if (
                PreviousState.piece_at(square) != CurrentState.piece_at(square) and
                (not CurrentState.piece_at(square) or CurrentState.piece_at(square).color != color)
            )
        ]

        return differences 
    
    def get_current_state(self):
        return self.States[self.state_counter - 1]
     
    def display_states(self):
        # Display all elements in the list
        for states in self.States:
            print(states)
            print("---------------------------")
            
            # initialized_board_white = chess.Board()
         
            # f8_square = chess.F8
            # black_rook = black_rook = chess.Piece(chess.ROOK, chess.BLACK)
            # print(board)
            
            # if board.status() != Status.VALID:
            #     print("Board is not valid")
            
            # board.push(chess.Move.from_uci("e2e4"))

            # if initialized_board_white == board:
            #     print("Boards are equal.")
            # else:
            #     print("Boards are not equal.")
            
            # differences = list(board.piece_map().items() - initialized_board_white.piece_map().items())
            
            # # Print the differences
       
            
            
            #print(initialized_board_white.piece_at(chess.E4))

            # print("board")
            # print(board)
            
            # print("E2")
            # print(chess.E2)
            
            # print(board.piece_at(chess.E2))
            
            # board.push(chess.Move.from_uci("e2e4"))
            # board.push(chess.Move.from_uci("e7e6"))
            
            # move = chess.Move.from_uci("f1g2")
            
            # if move in board.legal_moves:
            #     print(f"The move {move.uci()} is legal.")
            # else:
            #     print(f"The move {move.uci()} is not legal.")
            # #The variety of the moves that one piece is have 
            
            
            # square = chess.C3  # For example, the C3 square
            
            # piece_moves = board.legal_moves.count()
            
            # print(piece_moves)