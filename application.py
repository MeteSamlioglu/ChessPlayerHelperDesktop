from Modules import DetectBoard
from Modules import ClassifyOccupancy
from Modules import ClassifyPieces
from Modules import ChessStateTracker
import numpy as np
import chess
from chess import Status
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import functools
import cv2
import argparse
import typing
from recap import URI, CfgNode as CN
import matplotlib.pyplot as plt
from matplotlib import gridspec
from timeit import default_timer as timer
from pathlib import Path
import threading
import asyncio
import concurrent.futures
import time
import chess.engine
from stockfish import Stockfish
import threading
from queue import Queue
import imutils 

_squares = list(chess.SQUARES)

boardDetection = False
state_tracker = ChessStateTracker.StateTracker()
start_frame_ = None 

PREDICTION_THRESHOLD = 4

class BoardAnalyzer:
    
    def __init__(self, src_path: Path = URI("C:\\Users\\Monster\\Desktop\\Graduation Project1\\ChessPlayerHelperDesktop\\models\\occupancy_classifier")):
        self.cfg = CN.load_yaml_with_base("C:\\Users\\Monster\\Desktop\\Graduation Project1\\ChessPlayerHelperDesktop\\configuration\\corner_detection.yaml")
        
        self.occupancy_model, self.occupancy_cfg = ClassifyOccupancy.set_occupancy_classifier(
                                    Path("C:\\Users\\Monster\\Desktop\\Graduation Project1\\ChessPlayerHelperDesktop\\models\\occupancy_classifier\\"))
        
        self.occupancy_transforms_ = ClassifyOccupancy.build_transforms(self.occupancy_cfg)
        
        self.pieces_model, self.pieces_cfg = ClassifyPieces.set_piece_classifier(
                                    Path("C:\\Users\\Monster\\Desktop\\Graduation Project1\\python-scripts\\chesscog\\runs\\transfer_learning\\piece_classifier"))
        self.pieces_transforms_ =  ClassifyPieces.build_piece_transforms(self.pieces_cfg)
        
        self.piece_classes = np.array(list(map(ClassifyPieces.name_to_piece, self.pieces_cfg.DATASET.CLASSES)))
        
        self.stockfish = Stockfish(r'D:\\chesscog\\stockfish\\stockfish-windows-x86-64-avx2')

        self.moves = []
        
        self.square_coordinates = None
    
        self.CurrentFenDescription = None
        
        self.bestMove = None
        
    def detect_chessboard(self, frame) -> typing.Tuple[chess.Board, np.ndarray, dict]:
            
            #img = cv2.imread(path)
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # img, img_scale = DetectBoard.resize_image(self.cfg, img)
            
            detection_result, self.corners = DetectBoard.find_corners(self.cfg, img)
            
            self.square_coordinates = self.find_square_coordinates(self.corners)
            
            return detection_result, img, self.corners
    
    def predict_state(self, frame):
        
        with torch.no_grad():
                        
            occupancy_classification = ClassifyOccupancy.occupancy_classifier(self.occupancy_model, self.occupancy_transforms_,self.occupancy_cfg,
                                                                              frame, chess.WHITE, self.corners)            
            piece_predictions, pieces =  ClassifyPieces._classify_pieces(self.pieces_model,self.pieces_transforms_, self.piece_classes, frame, chess.WHITE, self.corners, occupancy_classification)

            # for piece_info in low_confidence_pieces:
            #     print(f"Square {piece_info['square']}: Predicted piece {piece_info['predicted_label']} with confidence {piece_info['confidence']}")

            state_tracker_result = state_tracker.add_state(piece_predictions, pieces)
                    
            return state_tracker_result
    
    def find_square_coordinates(self, corners) -> np.ndarray:
        
        corner1 = tuple(map(int, corners[0]))
        corner2 = tuple(map(int, corners[1]))
        corner3 = tuple(map(int, corners[2]))
        corner4 = tuple(map(int, corners[3]))            
        
        # Divide the line between corner2 and corner3 into 8 equal divisions
    
        num_divisions = 16
        
        division_points = [(corner1[0] + i * (corner2[0] - corner1[0]) / num_divisions,
                            corner1[1] + i * (corner2[1] - corner1[1]) / num_divisions)
                        for i in range(1, num_divisions, 2)]
        
        division_points_ = [(corner4[0] + i * (corner3[0] - corner4[0]) / num_divisions,
                            corner4[1] + i * (corner3[1] - corner4[1]) / num_divisions)
                        for i in range(1, num_divisions, 2)]
        
        # Divide the distance between corresponding points into 8 equal parts            
        point_to_take = 0

        square_coordinates = np.zeros((8, 8, 2))
        row = 7
        col = 0

        for point1, point2 in zip(division_points, division_points_):
            
            point_to_take = 0
            
            row = 7
            for alpha in np.linspace(0, 1, num_divisions + 2)[1:-1]:
                
                                    
                new_point = (
                    int((1 - alpha) * point1[0] + alpha * point2[0]),
                    int((1 - alpha) * point1[1] + alpha * point2[1])
                )
                                                        
                point_to_take+=1
                                    
                if point_to_take in [1, 2, 4, 6, 8, 10, 13, 16]:
                                        
                    square_coordinates[row][col] = new_point
                    row-=1                      
            col+=1
        
        return square_coordinates

    
    def get_square_coordinate(self, square_str, square_coordinates):
        
        square= chess.parse_square(square_str)

        column = chess.square_file(square)
        
        row = chess.square_rank(square)
                
        return square_coordinates[row][column]
        
    def show_move_on_board(self,frame, from_square, to_square, square_coordinates, offeredMove = True):
        
        from_square  = self.get_square_coordinate(from_square, square_coordinates)
        
        to_square = self.get_square_coordinate(to_square, square_coordinates)
        
        from_square = tuple(map(int, from_square))
        
        to_square = tuple(map(int, to_square))
        
        if offeredMove:
            cv2.arrowedLine(frame, from_square, to_square, (0, 128, 0), 5, cv2.LINE_AA, tipLength=0.1)
        
        else:
            cv2.arrowedLine(frame, from_square, to_square, (0, 255, 255), 5, cv2.LINE_AA, tipLength=0.1)
    
    def is_game_over(self,fen):
        board = chess.Board(fen)
        return board.is_game_over()
    
    def get_winner(self, fen):
        
        board = chess.Board(fen)
        result = board.result()
        print(result)
        if result == "1-0":
            return "White wins"
        elif result == "0-1":
            return "Black wins"
        elif result == "1/2-1/2":
            return "It's a draw"
    
    def rewind_to_previous_state(self):
        rewind_result = state_tracker.remove_last_state()
        if rewind_result == False:
            print("The rewind operation is not succeded")
    
    def analyze_board(self, frame):
        
        global boardDetection
        global start_frame_
        #frame = cv2.imread(path)
        
        result, img, self.corners = self.detect_chessboard(frame)
        
        if result == False:
            fail_corners = np.zeros((4,2))
            print("Board is not recognized !")
            boardDetection = False
            return fail_corners

        prediction_result, board, move = self.predict_state(img)
        
        if state_tracker.get_state_counter() == 1 and prediction_result: #Save the start_frame
            start_frame_ = frame    
            start_frame_ =  cv2.resize(start_frame_, (1200, 675))
            start_frame_ = cv2.cvtColor(start_frame_, cv2.COLOR_BGR2GRAY)
            start_frame_ = cv2.GaussianBlur(start_frame_, (21, 21), 0)
        
        if prediction_result == True and move is not None:
    
            self.moves.append(move)
            print(f'Move: {move}')
            
            self.CurrentFenDescription = board.fen()
            
            print("FEN Description")
            print(self.CurrentFenDescription)
            
            #Check if the game is over or not
            if self.is_game_over(self.CurrentFenDescription): 
                print("Checkmate!")
                print(self.get_winner(self.CurrentFenDescription))
            
            else:  #If the game is not over, analyze the board
                self.stockfish.set_fen_position(self.CurrentFenDescription)
            
                best_move = self.stockfish.get_best_move()
                
                if best_move:
                    self.BestMove = best_move
                else:
                    print("An issue occurred while calculating the suggested move.")
                
        
        boardDetection = True    

        return  self.corners
    """
     Returns the corners points
    
    """
    def get_corner_points(self):
        return self.corners
    
    def show_previous_move(self,frame):
        if len(self.moves) > 1:
            prev_move_index = len(self.moves) - 1 
            previous_move = self.moves[prev_move_index]
            
            from_square_name = chess.square_name(previous_move.from_square)
            
            to_square_name = chess.square_name(previous_move.to_square)
            
            self.show_move_on_board(frame, from_square_name, to_square_name, self.square_coordinates, False)
            
    def show_analyzed_move(self, frame):

            best_move =  self.BestMove
            
            from_square_name = best_move[:2]
            
            to_square_name = best_move[2:]

            self.show_move_on_board(frame, from_square_name, to_square_name, self.square_coordinates)

#------------------------------------------------------------------



def display_frames():
    global boardDetection
    global start_frame_
    
    ip_camera_url = f"http://10.251.108.229:8080/video"
    frame_counter = 0
    cap = cv2.VideoCapture(ip_camera_url)
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open IP camera stream at {ip_camera_url}")
        return

    width = 1200
    height = 675
    
    cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frames", width, height)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    boardAnalyzer = BoardAnalyzer() 

    future = None  
    corners = np.zeros((10,10))
    showBorders = False
    showPreviousMove = False
    showAnalyzedMove = False
    showMenuBar = False    
    alarm_counter = 0
    is_motion_detected = False
    enableMotionTracking = False
    text_motion_tracking = "5-)Enable Motion Tracking"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1200, 675))
        
        key = cv2.waitKey(1) & 0xFF

        if key== ord('b') and future is None:
            print("State is changed...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(boardAnalyzer.analyze_board, frame)        
        
        elif key == ord('a'):
            print("State is changed...")
            processing_thread = threading.Thread(target=boardAnalyzer.analyze_board, args=(frame,))
            processing_thread.start()
        elif key == ord('s'):
            if showMenuBar:
                showMenuBar = False
            else:
                showMenuBar = True            
        elif key == ord('4'):
            if showBorders:
                showBorders = False
            else:
                showBorders = True 
        elif key == ord('1'):
            if showPreviousMove:
                showPreviousMove = False
            else:
                showPreviousMove = True 
        elif key == ord('2'):
            if showAnalyzedMove:
                showAnalyzedMove = False
            else:
                showAnalyzedMove = True 

        elif key == ord('3'):
            boardAnalyzer.rewind_to_previous_state()
                  
        
        elif key == ord('5'):
            if enableMotionTracking:
                enableMotionTracking = False
                print("5Motion tracking is off")
                text_motion_tracking = "5-)Enable Motion Tracking"

            else:
                enableMotionTracking = True
                print("Motion tracking is on")
                text_motion_tracking = "5-)Disable Motion Tracking"
        
        elif key == ord('6'):
            print("ChessPlayerHelper is terminated")
            break  

        if enableMotionTracking:
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_bw = cv2.GaussianBlur(frame_bw, (5,5), 0)
            difference = cv2.absdiff(frame_bw, start_frame_)
            threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
            start_frame_ = frame_bw
            
            if threshold.sum() > 30:
                alarm_counter +=1
            else:
                if alarm_counter > 0:
                    alarm_counter -= 1
            if alarm_counter > 20:
                #print("Motion is detected")
                is_motion_detected = True
                    
            if alarm_counter == 0 and is_motion_detected:
                print("State is changed...")
                processing_thread = threading.Thread(target=boardAnalyzer.analyze_board, args=(frame,))
                processing_thread.start()
                is_motion_detected = False
    
        if boardDetection:
            corners = boardAnalyzer.get_corner_points()
            corner1 = tuple(map(int, corners[0]))
            corner2 = tuple(map(int, corners[1]))
            corner3 = tuple(map(int, corners[2]))
            corner4 = tuple(map(int, corners[3]))
            boardDetection = False
            
            if future is not None:
                future = None
 
        if showBorders == True:
            # Draw lines using the converted corners
            cv2.line(frame, corner1, corner2, (0, 255, 0), 10)
            cv2.line(frame, corner2, corner3, (0, 255, 0), 10)
            cv2.line(frame, corner3, corner4, (0, 255, 0), 10)
            cv2.line(frame, corner4, corner1, (0, 255, 0), 10)
 
        if showPreviousMove == True:
            boardAnalyzer.show_previous_move(frame)
        
        if showAnalyzedMove == True:
            boardAnalyzer.show_analyzed_move(frame)
        
        if showMenuBar == True:
            text = "Press 'a' to change state"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)  
            font_thickness = 2
            text_position = (10, 30)
            text2_position = (10, text_position[1] + 30)  
            text2 = "1-)Show previous move"
            text3_position = (10, text2_position[1] + 30) 
            text3 = "2-)Show analyzed move"
            text4 = "3-)Rewind to previous move"
            text4_position = (10, text3_position[1] + 30)
            text5 = "4-)Show detected board"
            text5_position = (10, text4_position[1] + 30)
            text6_position = (10, text5_position[1] + 30)
            text7 = "6-) Quit"
            text7_position = (10, text6_position[1] + 30)
            
            
            cv2.putText(frame, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text2, text2_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text3, text3_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text4, text4_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text5, text5_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text_motion_tracking, text6_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(frame, text7, text7_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        cv2.imshow("Frames", frame)

    cap.release()
    cv2.destroyAllWindows()

    # if boardDetection and future is not None:
    #     corners = future.result()
    #     corner1 = tuple(map(int, corners[0]))
    #     corner2 = tuple(map(int, corners[1]))
    #     corner3 = tuple(map(int, corners[2]))
    #     corner4 = tuple(map(int, corners[3]))
    #     boardDetection = False
    #     future = None


if __name__ == "__main__":
    # boardAnalyzer = BoardAnalyzer() 
    # boardAnalyzer.analyze_board()


    #state_tracker.display_states()
   
    display_frames()