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

_squares = list(chess.SQUARES)

boardDetection = False

state_tracker = ChessStateTracker.StateTracker()

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
        
    
    
    def detect_chessboard(self, frame) -> typing.Tuple[chess.Board, np.ndarray, dict]:
      
            
            #img = cv2.imread(path)
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img, img_scale = DetectBoard.resize_image(self.cfg, img)
            
            self.corners = DetectBoard.find_corners(self.cfg, img)
            
            return img, self.corners
    
    def predict_state(self, frame):
        
        with torch.no_grad():
                        
            occupancy_classification = ClassifyOccupancy.occupancy_classifier(self.occupancy_model, self.occupancy_transforms_,self.occupancy_cfg,
                                                                              frame, chess.WHITE, self.corners)
            #print(occupancy_classification)
            
            pieces =  ClassifyPieces._classify_pieces(self.pieces_model,self.pieces_transforms_, self.piece_classes, frame, chess.WHITE, self.corners, occupancy_classification)

            
            board = chess.Board()
            
            board.clear_board()
        
            for square, piece in zip(_squares, pieces):
                if piece:
                    #print(f'Square  {square} , Piece {piece} Type {type(piece)}')
                    board.set_piece_at(square, piece)
            
            
            state_tracker_result = state_tracker.add_state(board)
                    
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
        
        print(f'row {row} column {column}')
        
        return square_coordinates[row][column]
        
    def show_move_on_board(self,frame, from_square, to_square, square_coordinates):
        
        from_square  = self.get_square_coordinate(from_square, square_coordinates)
        
        to_square = self.get_square_coordinate(to_square, square_coordinates)
        
        from_square = tuple(map(int, from_square))
        
        to_square = tuple(map(int, to_square))
        
        cv2.arrowedLine(frame, from_square, to_square, (0, 128, 0), 10, cv2.LINE_AA, tipLength=0.2)
        
    def analyze_board(self, path):
        
        global boardDetection

        frame = cv2.imread(path)
        
        img, self.corners = self.detect_chessboard(frame)
        
        square_coordinates = self.find_square_coordinates(self.corners) 
        
        self.show_move_on_board(img, "d7", "d5", square_coordinates)
        
        cv2.imshow("Frame", img)

        cv2.waitKey(0)
        
        # counter = 0
        
        # prediction_result = self.predict_state(img)
        
        # while prediction_result is not True:
            
        #     if counter == PREDICTION_THRESHOLD:
        #         print("========= State is not recognized ===============")
        #         prediction_result = False
        #         break
                
        #     prediction_result = self.predict_state(img)
            
        #     counter+=1
       
        # if prediction_result:
        #     current_state_ = state_tracker.get_current_state()
        #     # print(current_state_)
        #     # print("---------------------------")
        
        
        # boardDetection = True    

        # return prediction_result

def display_frames():
    global boardDetection
    ip_camera_address = '192.168.1.85'
    ip_camera_port = '8080'
    ip_camera_url = f"http://192.168.1.85:8080/video"
    frame_counter = 0
    cap = cv2.VideoCapture(ip_camera_url)
    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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

    frame_counter = 0
    future = None  
    corners = np.zeros((10,10))
    showBorders = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1200, 675))
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        if cv2.waitKey(25) & 0xFF == ord('a') and future is None:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(boardAnalyzer.analyzeBoard, frame)        
        if cv2.waitKey(25) & 0xFF == ord('s'):
            if showBorders:
                showBorders = False
            else:
                showBorders = True 
                
        if boardDetection and future is not None:
            corners = future.result()
            corner1 = tuple(map(int, corners[0]))
            corner2 = tuple(map(int, corners[1]))
            corner3 = tuple(map(int, corners[2]))
            corner4 = tuple(map(int, corners[3]))
            boardDetection = False
            future = None
 
        if showBorders == True:
            # Draw lines using the converted corners
            cv2.line(frame, corner1, corner2, (0, 255, 0), 10)
            cv2.line(frame, corner2, corner3, (0, 255, 0), 10)
            cv2.line(frame, corner3, corner4, (0, 255, 0), 10)
            cv2.line(frame, corner4, corner1, (0, 255, 0), 10)
            start_point = corner2
            end_point = corner4
            color = (255, 0, 0)  # Green color
            thickness = 10
            arrow_tip_length = 0.1

            # Draw the arrowed line
            #cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=arrow_tip_length)

        cv2.imshow("Frames", frame)

    
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    boardAnalyzer = BoardAnalyzer() 
    #boardAnalyzer.analyzeBoard()
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\00.jpg")
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\01.jpg")
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\02.jpg")
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\03.jpg")
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\04.jpg")
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\05.jpg")
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\06.jpg")
    boardAnalyzer.analyze_board("D:\\chesscog\\data\\white_states\\07.jpg")





    #state_tracker.display_states()

   
    #display_frames()