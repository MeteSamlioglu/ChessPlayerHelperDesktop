from Modules import DetectBoard
from Modules import ClassifyOccupancy
from Modules import ClassifyPieces
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
        
    
    
    def analyzeBoard(self, frame) -> typing.Tuple[chess.Board, np.ndarray, dict]:
        with torch.no_grad():
            global boardDetection 
            
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img, img_scale = DetectBoard.resize_image(self.cfg, img)
            
            inverse_matrix, corners = DetectBoard.find_corners(self.cfg, img)
            
            img_shape = img.shape
            height, width = img_shape[:2]
            #print(f'"Image size: Width={width}, Height={height}')
            # for point in corners:
            #     cv2.circle(img, tuple(point.astype(int)), 5, (0, 0, 255), -1)  # Convert point to integers
          
           
            
            # point_ = tuple((corners[0][0].astype(int) + 225, corners[0][1].astype(int) - 12))
            
            #cv2.circle(img, point_, 5, (0, 0, 255), -1)  # Convert point to integers

            # cv2.imshow("img",img)
            # cv2.waitKey(0)
            
            occupancy_classification = ClassifyOccupancy.occupancy_classifier(self.occupancy_model, self.occupancy_transforms_,self.occupancy_cfg,
                                                                              img, chess.WHITE, corners)
            #print(occupancy_classification)
            
            pieces =  ClassifyPieces._classify_pieces(self.pieces_model,self.pieces_transforms_, self.piece_classes, img, chess.WHITE, corners, occupancy_classification)

            #print(pieces)
            board = chess.Board()
            board.clear_board()
       
            for square, piece in zip(_squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            
            #print(board)
            
           
            # if board.status() != Status.VALID:
            #     print("Board is not valid")
            
            # cropped_square_ = DetectBoard._warp_points(inverse_matrix, cropped_square)
            # cv2.imshow("cropped_square_", cropped_square_)
            # cv2.waitKey(0)
            boardDetection = True
            return corners

def display_frames():
    global boardDetection
    ip_camera_address = '192.168.1.62'
    ip_camera_port = '8080'
    ip_camera_url = f"http://192.168.1.62:8080/video"
    frame_counter = 0
    cap = cv2.VideoCapture(ip_camera_url)

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
            cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=arrow_tip_length)

        cv2.imshow("Frames", frame)

    
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #boardAnalyzer = BoardAnalyzer() 
    #boardAnalyzer.analyzeBoard()
    display_frames()