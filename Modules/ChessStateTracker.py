import chess
from chess import Status
import copy
_squares = list(chess.SQUARES)
import heapq

class StateTracker:
    
    THRESHOLD_VALUE = 8
    THRESHOLD_PREDICTION = 0.3
    def __init__(self):
        
        self.States = []
        
        self.StatePredictions = []
        self.States_piece_count = []
        self.state_counter = 0
        
        self.from_square = []
        self.to_square = []
        
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
            
            return CurrentState
        else: 
            
            previous_state = self.States[self.state_counter - 1]
            
            previous_predictions = self.StatePredictions[self.state_counter - 1]
        
            previous_predictions_dict = {d['square']: d for d in previous_predictions}
            piece_predictions_dict = {d['square']: d for d in piece_predictions}
           
      
            move_turn_for = chess.WHITE if self.WHITE_TURN == True else chess.BLACK 
            
            print("Difference")
            for square, current_piece_info in piece_predictions_dict.items():

                current_piece_confidence = current_piece_info['confidence']
                
                current_piece_label = current_piece_info['predicted_label']
                
                if square in previous_predictions_dict:
                    
                    previous_piece_info = previous_predictions_dict[square]
                    
                    previous_piece_confidence = previous_piece_info['confidence']
                    
                    if previous_piece_confidence > 0.8 and (previous_piece_confidence - current_piece_confidence) > StateTracker.THRESHOLD_PREDICTION:
                        self.from_square.append(previous_piece_info)
                else:
                    self.to_square.append(current_piece_info)
                
            if len(self.from_square) > 0:
                print("From square")
                print(self.from_square)
                for i in range(len(self.from_square)):
                    from_square_name = chess.square_name(self.from_square[i]['square'])
                    print(from_square_name)
            
            if len(self.to_square) > 0:
                print("To Square")
                print(self.to_square)    
                
                for i in range(len(self.to_square)):
                    to_square_name = chess.square_name(self.to_square[i]['square'])
                    print(to_square_name)

            counter = 0
            for square, piece in zip(_squares, pieces):
                if piece:
                    current_piece_info = piece_predictions[counter]
                    
                    current_piece_confidence = current_piece_info['confidence']            
                                                                            
                    CurrentState.set_piece_at(square, piece)
                    
                    counter+=1
       
            
            print("CurrentState")
            print(CurrentState)
            print("PreviousState")
            print(previous_state)
                                
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
               self.StatePredictions.append(piece_predictions)
               
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
            
          
            
            print("White possible moves")
            
            if len(white_possible_moves) == 0:
                possible_moves_from_prediction_threshold = []
                if len(self.from_square) > 0 and len(self.to_square) > 0:
                    for i in range(len(self.from_square)):
                        from_square_name = chess.square_name(self.from_square[i]['square'])
                        for j in range(len(self.to_square)):
                            to_square_name = chess.square_name(self.to_square[j]['square'])
                                     
                            from_square = chess.parse_square(from_square_name)
                            
                            to_square = chess.parse_square(to_square_name)
                            
                            move = chess.Move(from_square, to_square)                            
                            
                            previous_state.turn = chess.WHITE
                            
                            is_legal_move = previous_state.is_legal(move)
                            if is_legal_move:
                                possible_moves_from_prediction_threshold.append(move)
                    
                    if len(possible_moves_from_prediction_threshold) > 0:
                        white_possible_moves = possible_moves_from_prediction_threshold
                
            
            if len(white_possible_moves) == 0:
                return False, CurrentState
            
            print(white_possible_moves)
            
            current_state =  chess.Board()
            current_state.clear()
            
            for square in chess.SQUARES:
                piece = previous_state.piece_at(square)
                if piece is not None:
                    current_state.set_piece_at(square, piece)
            
            index = 0
            
            if(len(white_possible_moves) > 1):
                index = self.compare_detected_pieces(white_possible_moves, CurrentState, previous_state)
          
            
            move = white_possible_moves[index]
            
            current_state.push(move)
            
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
            
            self.StatePredictions.append(piece_predictions)
            
            return True, current_state
        
        if self.BLACK_TURN == True:
        
            
            black_difference_set = self.find_difference(previous_state, CurrentState, chess.WHITE)
            
            black_diff_count = len(black_difference_set)
            
            print(black_diff_count)
            # print(black_difference_set)

            black_possible_moves = self.find_possible_moves(black_difference_set, chess.BLACK)
            
            if black_diff_count >= StateTracker.THRESHOLD_VALUE: 
                return False
            

            print("Black possible moves")
            
            if len(black_possible_moves) == 0:
                
                possible_moves_from_prediction_threshold = []
                if len(self.from_square) > 0 and len(self.to_square) > 0:
                    for i in range(len(self.from_square)):
                        from_square_name = chess.square_name(self.from_square[i]['square'])
                        for j in range(len(self.to_square)):
                            to_square_name = chess.square_name(self.to_square[j]['square'])
                                     
                            from_square = chess.parse_square(from_square_name)
                            
                            to_square = chess.parse_square(to_square_name)
                            
                            move = chess.Move(from_square, to_square)                            
                            
                            previous_state.turn = chess.BLACK
                            
                            is_legal_move = previous_state.is_legal(move)
                            if is_legal_move:
                                possible_moves_from_prediction_threshold.append(move)
                    
                    if len(possible_moves_from_prediction_threshold) > 0:
                        black_possible_moves = possible_moves_from_prediction_threshold
            
            
            if len(black_possible_moves) == 0:
                return False, CurrentState    
               
            print(black_possible_moves)
            current_state =  chess.Board()
            current_state.clear()
            
            for square in chess.SQUARES:
                piece = previous_state.piece_at(square)
                if piece is not None:
                    current_state.set_piece_at(square, piece)
            
            index = 0
            if(len(black_possible_moves) > 1):
                index = self.compare_detected_pieces(black_possible_moves, CurrentState, previous_state)
            
            move = black_possible_moves[index]

            current_state.turn = chess.BLACK
            current_state.push(move)
      
            self.state_counter+=1

            print(f'State Number {self.state_counter}')
            print(f'Turn White = {self.BLACK_TURN}')
            print("Current State Black")
            print(current_state)
            print("---------------------------")   
            
            self.WHITE_TURN = True
            
            self.BLACK_TURN = False   
            
            current_piece_count = self.get_piece_count(current_state)
            
            self.States_piece_count.append(current_piece_count)
            
            self.StatePredictions.append(piece_predictions)
            
            print(f'Current piece count on the board {current_piece_count}')

            self.States.append(current_state)
            
            return True, current_state
    
    def find_possible_moves_prediction_threshold(self, from_square, to_square, turn : chess.Color):
        print("Sa")
    
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
    
    
    def compare_detected_pieces(self, possible_moves, CurrentState, previous_state):
        
        index = 0      
        
        for i in range(len(possible_moves)):
            move = possible_moves[i].uci()
            
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
                    
        diff_squares = []
            
        legal_moves = []    
        
        print("Difference Set")
        print(difference_set)
        
        previous_board = self.States[self.state_counter - 1] #Get the previous state
        
        for square, current_piece, previous_piece in difference_set:            
            diff_squares.append(chess.square_name(square))
        
        if turn == chess.BLACK:

            for i in range(len(diff_squares)):
                
                square= chess.parse_square(diff_squares[i])
                
                piece = previous_board.piece_at(square)
                print("Difference black")
                square_name = chess.square_name(square)

                print(f'Piece {piece} , Square ({square_name})')
                
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
                square_name = chess.square_name(square)
                print(f'Piece {piece} , Square ({square_name})')
                     
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
            
            
            
            
            
            
            # if piece_counter > previous_board_piece_count:
                
            #     print(f'Current piece count {previous_board_piece_count} Found piece counter {piece_counter}')  
                
            #     n = piece_counter - previous_board_piece_count
                
            #     max_heap = []

            #     for piece_info in piece_predictions:
            #         confidence = piece_info['confidence']
            #         heapq.headppush(max_heap, (-confidence, piece_info))

            #         if len(max_heap) > n:
            #             heapq.heappop(max_heap)

            #     # Retrieve and print the lowest N elements
            #     for neg_confidence, piece_info in max_heap:
                    
            #         square = piece_info['square']
            #         confidence = -neg_confidence

            #         print(f"Square: {piece_info['square']}, Label: {piece_info['predicted_label']}, Confidence: {confidence}")
            #         CurrentState.set_piece_at(square, None)