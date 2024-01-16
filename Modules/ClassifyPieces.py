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
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2
MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = .25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])
_squares = list(chess.SQUARES)

def set_piece_classifier(path: Path):
    
    piece_model_path = next(iter(path.glob("*.pt")))
    
    piece_yaml_file = next(iter(path.glob("*.yaml")))
    
    configuration_file = CN.load_yaml_with_base(piece_yaml_file)
    
    model = torch.load(piece_model_path, map_location = DEVICE)

    model = device(model)
    
    model.eval()
    
    return model, configuration_file


def build_piece_transforms(cfg: CN) -> typing.Callable:

    transforms = cfg.DATASET.TRANSFORMS

    t = []
    if transforms.CENTER_CROP:
        t.append(T.CenterCrop(transforms.CENTER_CROP))

    if transforms.RESIZE:
        t.append(T.Resize(tuple(reversed(transforms.RESIZE))))
    t.extend([T.ToTensor(),
              T.Normalize(mean=_MEAN, std=_STD)])
    return T.Compose(t)

def name_to_piece(name: str) -> chess.Piece:
    """Convert the name of a piece to an instance of :class:`chess.Piece`.

    Args:
        name (str): the name of the piece

    Returns:
        chess.Piece: the instance of :class:`chess.Piece`
    """
    color, piece_type = name.split("_")
    color = color == "white"
    piece_type = chess.PIECE_NAMES.index(piece_type)
    return chess.Piece(piece_type, color)


def _classify_pieces(_pieces_model, _pieces_transforms, _piece_classes, img: np.ndarray, turn: chess.Color, corners: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    occupied_squares = np.array(_squares)[occupancy]

    
    warped = WarpBoardImage(img, corners)
    
    #piece_coordinates = self._detect_pieces_and_print_coordinates(img, turn, corners, occupancy)

    piece_imgs = map(functools.partial(
        crop_square, warped, turn=turn), occupied_squares)
    piece_imgs = map(Image.fromarray, piece_imgs)
    piece_imgs = map(_pieces_transforms, piece_imgs)
    piece_imgs = list(piece_imgs)
    piece_imgs = torch.stack(piece_imgs)
    piece_imgs = device(piece_imgs)
    pieces = _pieces_model(piece_imgs)
    pieces = pieces.argmax(axis=-1).cpu().numpy()
    pieces = _piece_classes[pieces]
    all_pieces = np.full(len(_squares), None, dtype=object)
    all_pieces[occupancy] = pieces
    
    logits = _pieces_model(piece_imgs)

    probabilities = F.softmax(logits, dim=-1)

    # Get the predicted classes
    predicted_classes = probabilities.argmax(axis=-1).cpu().numpy()

    # Get the class labels
    predicted_labels = _piece_classes[predicted_classes]

    # Get the confidence scores for each prediction
    confidence_scores = probabilities.max(axis=-1).values.cpu().numpy()
    
    pieces_info = []
    
    for square, label, confidence in zip(occupied_squares, predicted_labels, confidence_scores):
        piece_info = {
            'square': square,
            'predicted_label': label,
            'confidence': confidence
        }
        pieces_info.append(piece_info)
        
    return pieces_info, all_pieces
    
    
def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:

    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    height_increase = MIN_HEIGHT_INCREASE + \
        (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7)
    left_increase = 0 if col >= 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3)
    right_increase = 0 if col < 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3)
    x1 = int(MARGIN + SQUARE_SIZE * (col - left_increase))
    x2 = int(MARGIN + SQUARE_SIZE * (col + 1 + right_increase))
    y1 = int(MARGIN + SQUARE_SIZE * (row - height_increase))
    y2 = int(MARGIN + SQUARE_SIZE * (row + 1))
    width = x2-x1
    height = y2-y1
    cropped_piece = img[y1:y2, x1:x2]
    if col < 4:
        cropped_piece = cv2.flip(cropped_piece, 1)
    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped_piece.dtype)
    result[OUT_HEIGHT - height:, :width] = cropped_piece
    
    return result

def WarpBoardImage(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image of the chessboard onto a regular grid.

    Args:
        img (np.ndarray): the image of the chessboard
        corners (np.ndarray): pixel locations of the four corner points

    Returns:
        np.ndarray: the warped image
    """
    src_points = sortCornerPoints(corners)
    
    dst_points = np.array([[MARGIN, MARGIN],  # top left
                            [BOARD_SIZE + MARGIN, MARGIN],  # top right
                            [BOARD_SIZE + MARGIN, \
                                BOARD_SIZE + MARGIN],  # bottom right
                            [MARGIN, BOARD_SIZE + MARGIN]  # bottom left
                            ], dtype=np.float32)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))



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

