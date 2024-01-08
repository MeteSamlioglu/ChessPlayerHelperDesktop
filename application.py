from Modules import DetectBoard
from Modules import ClassifyOccupancy
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

class BoardAnalyzer:
    
    def __init__(self, src_path: Path = URI("C:\\Users\\Monster\\Desktop\\Graduation Project1\\ChessPlayerHelperDesktop\\models\\occupancy_classifier")):
        self.cfg = CN.load_yaml_with_base("C:\\Users\\Monster\\Desktop\\Graduation Project1\\ChessPlayerHelperDesktop\\configuration\\corner_detection.yaml")
        
        self.occupancy_model, self.occupancy_cfg = ClassifyOccupancy.set_occupancy_classifier(
                                    Path("C:\\Users\\Monster\\Desktop\\Graduation Project1\\ChessPlayerHelperDesktop\\models\\occupancy_classifier\\"))
        
        self.occupancy_transforms_ = ClassifyOccupancy.build_transforms(self.occupancy_cfg)
    
    def analyzeBoard(self) -> typing.Tuple[chess.Board, np.ndarray, dict]:
        with torch.no_grad():
            
            img = cv2.imread("D:\\chesscog\\example\\myboard4.jpg")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img, img_scale = DetectBoard.resize_image(self.cfg, img)
            
            corners = DetectBoard.find_corners(self.cfg, img)
            
            print(corners.shape)
            
            for point in corners:
                cv2.circle(img, tuple(point.astype(int)), 5, (0, 0, 255), -1)  # Convert point to integers
            
            cv2.imshow("img",img)
            cv2.waitKey(0)
            
            occupancy_classification = ClassifyOccupancy.occupancy_classifier(self.occupancy_model, self.occupancy_transforms_,self.occupancy_cfg,
                                                                              img, chess.WHITE, corners)
            print(occupancy_classification)
            
            # occupancy_classification = np.array(occupancy_classification)
            # for i in range(occupancy_classification.shape[0]):
            #     if i % 8 != 0:
            #         print(f'{occupancy_classification[i]}', end=' ')
            #     else:
            #         print("\n")

if __name__ == "__main__":
    boardAnalyzer = BoardAnalyzer() 
    boardAnalyzer.analyzeBoard()