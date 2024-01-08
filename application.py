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



def analyzeBoard()-> typing.Tuple[chess.Board, np.ndarray, dict]:
    
    #corner_detection_configuration_file_path = URI("Configuration://corner_detection.yaml")

    cfg = CN.load_yaml_with_base("C:\\Users\\Monster\\Desktop\\Graduation Project1\\chesscog\\configuration\\corner_detection.yaml")

    with torch.no_grad():
        
        img = cv2.imread("D:\\chesscog\\example\\myboard4.jpg")
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img, img_scale = DetectBoard.resize_image(cfg, img)
        
        corners = DetectBoard.find_corners(cfg,img)
        
        classify_occupancy = ClassifyOccupancy.OccupancyClassifier()
        
        occupancy_classification = classify_occupancy.occupancy_classifier(img, chess.WHITE ,corners)
        
        print(occupancy_classification)
        # occupancy_classification = np.array(occupancy_classification)
        # for i in range(occupancy_classification.shape[0]):
        #     if i % 8 != 0:
        #         print(f'{occupancy_classification[i]}', end = ' ')
        #     else:
        #         print("\n")
    
def predict(setup : callable = lambda : None):
    setup()
    
    
    
if __name__ == "__main__":
    analyzeBoard()