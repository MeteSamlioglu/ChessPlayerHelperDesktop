import numpy as np
import chess
from chess import Status
from pathlib import Path
from recap import URI, CfgNode as CN
import torch
import torch.optim as optim
import chess
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Core.device import device, DEVICE
import cv2
import functools
from PIL import Image
import typing

SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE
        
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])

_squares = list(chess.SQUARES)



def build_transforms(cfg: CN) -> typing.Callable:

    transforms = cfg.DATASET.TRANSFORMS

    t = []
    if transforms.CENTER_CROP:
        t.append(T.CenterCrop(transforms.CENTER_CROP))

    if transforms.RESIZE:
        t.append(T.Resize(tuple(reversed(transforms.RESIZE))))
    t.extend([T.ToTensor(),
              T.Normalize(mean=_MEAN, std=_STD)])
    return T.Compose(t)
        
def set_occupancy_classifier(path: Path):
    
    file_path = next(iter(path.glob("*.pt")))
    
    yaml_file = next(iter(path.glob("*.yaml")))
    
    configuration_file = CN.load_yaml_with_base(yaml_file)
    
    model = torch.load(file_path, map_location = DEVICE)

    model = device(model)
    
    model.eval()
    
    return model, configuration_file


    
"""It arranges the board coordinates to the order of [TL, TR, BR, BL].
    
Param:
    corners: the edge coordinates of the chess board
Returns:
    Sorted coordinates according to TL, TR, BR , BL
""" 
def sortCornerPoints(corners : np.ndarray) -> np.ndarray:
    
    corners =  corners[corners[:, 1].argsort()] # Ordering y coordinates
    corners[:2] = corners[:2][corners[:2, 0].argsort()] #Only sort the top x-coordinates
    corners[2:] = corners[2:][corners[2:, 0].argsort()[::-1]] #Sort bottom x-coordinates
    return corners



def WarpBoardImage(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image of the chessboard onto a regular grid.

    Args:
        img (np.ndarray): the image of the chessboard
        corners (np.ndarray): pixel locations of the four corner points

    Returns:
        np.ndarray: the warped image
    """
    src_points = sortCornerPoints(corners)
    
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                        [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                        [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right
                        [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                        ], dtype=np.float32)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))


def occupancy_classifier(model, transforms_, config_file : CN, img: np.ndarray, turn: chess.Color, corners: np.ndarray):
    
    path = URI("C:\\Users\\Monster\\Desktop\\Graduation Project1\\chesscog\\models\\occupancy_classifier")
    # yaml_file = next(iter(path.glob("*.yaml")))
    # configuration_file = CN.load_yaml_with_base(yaml_file)

    warped_image = WarpBoardImage(img, corners) # Warp the image of the chessboard for cropping squares

    
    square_imgs = map(functools.partial(
        CropChessBoardSquares, warped_image, turn=turn), _squares)      
    
    square_imgs = map(Image.fromarray, square_imgs)

    square_imgs = map(transforms_, square_imgs)
    
    square_imgs = list(square_imgs)
    
    square_imgs = torch.stack(square_imgs)

    square_imgs = device(square_imgs)
    
    occupancy = model(square_imgs)
    
    occupancy = occupancy.argmax(
        axis=-1) == config_file.DATASET.CLASSES.index("occupied")  

    occupancy = occupancy.cpu().numpy()
    
    return occupancy
      
def CropChessBoardSquares(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    
    rank = chess.square_rank(square)
    file = chess.square_file(square)
  
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    
    cropped_square = img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
                        int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]
    
    return cropped_square
