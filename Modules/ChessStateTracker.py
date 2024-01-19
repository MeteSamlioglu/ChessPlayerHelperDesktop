import chess
from chess import Status
import copy
_squares = list(chess.SQUARES)
import heapq

class StateTracker:
    
    THRESHOLD_VALUE = 5
    
    def __init__(self):
        
        self.States = []
        self.States_piece_count = []
        self.state_counter = 0
        
        self.WHITE_TURN = True
        self.BLACK_TURN = False
        
        if self.WHITE_TURN == True:
            self.white_initialized_board = chess.Board()
        
    def load_the_board(self, piece_predictions, pieces):
        CurrentState = chess.Board()
        CurrentState.clear_board()
        
        if self.state_counter == 0:
            for square, piece in zip(_squares, pieces):
                if piece:
                    CurrentState.set_piece_at(square, piece)
            
            piece_counter = 32 #Initial board has 32 piece 
            return CurrentState
        else: 
            previous_state = self.States[self.state_counter - 1]
            counter = 0
            piece_counter = 0    

            for square, piece in zip(_squares, pieces):
                if piece:
                    piece_info = piece_predictions[counter]
                    
                    confidence = piece_info['confidence']            
                    
                    integer_square = square
                    algebraic_square = chess.square_name(integer_square)
                    
                    print(f'square {algebraic_square} piece {piece} confidence {confidence}')
                    
                    if confidence < 0.48: #0.48
                        CurrentState.set_piece_at(square, previous_state.piece_at(square)) 
                        piece_counter+=1
                    else:
                        CurrentState.set_piece_at(square, piece)
                        piece_counter+=1
                    
                    counter+=1
            
            print(CurrentState)
            previous_board_piece_count = self.States_piece_count[self.state_counter - 1]
            
            if piece_counter > previous_board_piece_count:
                
                print(f'Current piece count {previous_board_piece_count} Found piece counter {piece_counter}')  
                
                n = piece_counter - previous_board_piece_count
                
                max_heap = []

                for piece_info in piece_predictions:
                    confidence = piece_info['confidence']
                    heapq.heappush(max_heap, (-confidence, piece_info))

                    if len(max_heap) > n:
                        heapq.heappop(max_heap)

                # Retrieve and print the lowest N elements
                for neg_confidence, piece_info in max_heap:
                    
                    square = piece_info['square']
                    confidence = -neg_confidence

                    print(f"Square: {piece_info['square']}, Label: {piece_info['predicted_label']}, Confidence: {confidence}")
                    CurrentState.set_piece_at(square, None)
                    
            return CurrentState
        
    def add_state(self, piece_predictions, pieces):
        
        CurrentState = self.load_the_board(piece_predictions, pieces)
        
        if self.state_counter == 0:
            if CurrentState !=  self.white_initialized_board:
                print("======== Initial board is not recognized properly ========")
                return False, CurrentState
            else:
               print("======== Initial board is recognized  ========")
               self.States.append(self.white_initialized_board)
               initial_piece_count = 32
               self.States_piece_count.append(initial_piece_count)
               
               self.state_counter+=1
               
               print(f'State Number {self.state_counter}')
               print("CurrentState")
               print(CurrentState)
               
               return True, self.white_initialized_board
        
        previous_state = self.States[self.state_counter - 1]

        
        if self.WHITE_TURN == True:
                    
            white_differences_set = self.find_difference(previous_state, CurrentState, chess.BLACK)
            
            white_diff_count = len(white_differences_set)
            # print(white_differences_set)
            print(white_diff_count)
            
            white_possible_moves = self.find_possible_moves(white_differences_set, chess.WHITE)
            
            if white_diff_count >= StateTracker.THRESHOLD_VALUE: 
                return False
            
          
            possible_white_moves_count = len(white_possible_moves)
            
            print("White possible moves")
            
            
            if(possible_white_moves_count == 0):
                return False, CurrentState
            
            print(white_possible_moves)
            
            current_state =  chess.Board()
            current_state.clear()
            
            for square in chess.SQUARES:
                piece = previous_state.piece_at(square)
                if piece is not None:
                    current_state.set_piece_at(square, piece)
            
            # index = 0
            # if(len(white_possible_moves) > 1):
            #     index = self.compare_detected_pieces(white_possible_moves, CurrentState, previous_state)
          
            
            move = white_possible_moves[0]
            
            current_state.push(move)
            # print("Previous State White")
            # print(CurrentState)
            self.state_counter+=1
            
            print(f'State Number {self.state_counter}')
            print(f'Turn White = {self.WHITE_TURN}  {chess.WHITE}')
            print("Current State White")
            print(current_state)
            print("---------------------------")  
            
            self.WHITE_TURN = False
            
            self.BLACK_TURN = True       

            current_piece_count = self.get_piece_count(current_state)
            
            self.States_piece_count.append(current_piece_count)
            print(f'Current piece count on the board {current_piece_count}')
            
            self.States.append(current_state)
            
            return True, current_state
        
        if self.BLACK_TURN == True:
        
            
            black_difference_set = self.find_difference(previous_state, CurrentState, chess.WHITE)
            
            black_diff_count = len(black_difference_set)
            
            print(black_diff_count)
            # print(black_difference_set)

            black_possible_moves = self.find_possible_moves(black_difference_set, chess.BLACK)
            
            if black_diff_count >= StateTracker.THRESHOLD_VALUE: 
                return False
            
            possible_black_moves_count = len(black_possible_moves)

            print("Black possible moves")
            
            if possible_black_moves_count == 0:
                return False, CurrentState
            
            print(black_possible_moves)
            current_state =  chess.Board()
            current_state.clear()
            
            for square in chess.SQUARES:
                piece = previous_state.piece_at(square)
                if piece is not None:
                    current_state.set_piece_at(square, piece)
            
            index = 0
            # if(len(black_possible_moves) > 1):
            #     index = self.compare_detected_pieces(black_possible_moves, CurrentState, previous_state)
            
            move = black_possible_moves[index]

            current_state.turn = chess.BLACK
            current_state.push(move)
      
            self.state_counter+=1

            print(f'State Number {self.state_counter}')
            print(f'Turn White = {self.BLACK_TURN}  {chess.BLACK}')
            print("Current State Black")
            print(current_state)
            print("---------------------------")   
            
            self.WHITE_TURN = True
            
            self.BLACK_TURN = False   
            
            current_piece_count = self.get_piece_count(current_state)
            
            self.States_piece_count.append(current_piece_count)
            
            print(f'Current piece count on the board {current_piece_count}')

            self.States.append(current_state)
            
            return True, current_state
    
    def get_piece_count(self, chess_board):
        piece_count = 0
        
        # Iterate over the piece_map to count pieces
        for square, piece in chess_board.piece_map().items():
            if piece is not None:
                piece_count += 1

        return piece_count
    
    def isBlackTurn(self):
        if(self.BLACK_TURN == True):
            return True
        else:
            return False
    
    def compare_detected_pieces(self, white_possible_moves, CurrentState, previous_state):
        
        index = 0      
        
        for i in range(len(white_possible_moves)):
            move = white_possible_moves[i].uci()
            
            from_square = move[:2]
            
            to_square = move[2:]
            
            from_square = chess.parse_square(from_square)
            
            to_square = chess.parse_square(to_square)
            
            piece_at_previous_state = previous_state.piece_at(from_square)
        
            piece_at_current_state = CurrentState.piece_at(to_square)

            if(piece_at_previous_state == piece_at_current_state):
                index = i 
        
        return index    

    def find_possible_moves(self, difference_set, turn : chess.Color):
        
        diff_count = len(difference_set)
            
        diff_squares = []
            
        legal_moves = []    
        
        previous_board = self.States[self.state_counter - 1] #Get the previous state
        
        for square, current_piece, previous_piece in difference_set:            
            diff_squares.append(chess.square_name(square))
        
        if turn == chess.BLACK:

            for i in range(len(diff_squares)):
                
                square= chess.parse_square(diff_squares[i])
                
                piece = previous_board.piece_at(square)
                print("Difference black")
                print(f'Piece {piece} , Square ({square})')
                if piece is not None and piece.color == chess.BLACK:
         
                    for j in range(len(diff_squares)):
                        
                        if(j != i):    
                                                        
                            move_notation = f'{diff_squares[i]}{diff_squares[j]}'
               
                            from_square = chess.parse_square(move_notation[:2].lower())
                            
                            to_square = chess.parse_square(move_notation[2:].lower())
                            
                            move = chess.Move(from_square, to_square)                            
                            previous_board.turn = chess.BLACK

                            is_legal_move = previous_board.is_legal(move)
                            
                            if is_legal_move:
                                legal_moves.append(move)
            

        elif turn == chess.WHITE:
            
            previous_board.turn = chess.WHITE

            for i in range(len(diff_squares)):
                
                square= chess.parse_square(diff_squares[i])
                
                piece = previous_board.piece_at(square)
               
                print("Difference white")
                print(f'Piece {piece} , Square ({square})')
                     
                if piece is not None and piece.color == chess.WHITE:
               
                    for j in range(len(diff_squares)):
                        
                        if(j != i):    
                                                        
                            move_notation = f'{diff_squares[i]}{diff_squares[j]}'
                            
                            from_square = chess.parse_square(move_notation[:2].lower())
                            
                            to_square = chess.parse_square(move_notation[2:].lower())
                            
                            move = chess.Move(from_square, to_square)                            
                            
                            is_legal_move = previous_board.is_legal(move)
                            
                            if is_legal_move:
                                legal_moves.append(move)
            
            if(self.state_counter > 1):
                previous_board.turn = chess.BLACK          
        
        
        return legal_moves
        
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
            
            
            # print("Black's Turn")
            
            # print(f'Number diff moves {diff_count}')
            
            # print("Differences in Black Pieces:")
            
            # for square, current_piece, previous_piece in difference_set:
            #     print(f"At square {chess.square_name(square)}, current board has {current_piece} and previous board has {previous_piece}.")
            #     diff_squares.append(chess.square_name(square))
            
            # print("White's Turn")
            
            # print(f'Number diff moves {diff_count}')
            
            # print("Differences in White Pieces:")
            # for square, current_piece, previous_piece in difference_set:
            #     print(f"At square {chess.square_name(square)}, current board has {previous_piece} and previous board has {current_piece}.")
            