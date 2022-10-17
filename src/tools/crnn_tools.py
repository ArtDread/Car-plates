"""Module provides processing of image and logic for decoding crnn's output."""
from typing import Dict, List

import cv2
import numpy as np
import torch
from numpy.typing import NDArray


def process_image(image: NDArray[np.float32]) -> torch.Tensor:
    """
    Transform into tensor from numpy.

    :param image: image of cropped car plate, size (H, W, C)
    :returns: transformed image, size (1, C, H, W)
    """
    image = cv2.resize(image, (320, 64), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return img


def decode(pred_seq: NDArray[np.float32], idx_to_char: Dict[int, str]) -> str:
    """
    Convert predicted indexed tokens by crnn model to chars.

    :param pred_seq: crnn output, size (1, 18, 23)
    :param idx_to_char: vocabulary for decoding indexed tokens
    :returns: decoded car plate text sequence
    """
    pred_seq = pred_seq[0]
    # Consists of blank
    seq = []
    for i in range(len(pred_seq)):
        # Take max possible class in indexed form
        label_idx = np.argmax(pred_seq[i])
        # Convert to char
        seq.append(idx_to_char[label_idx])
    # Remove blanks
    out: List[str] = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != "blank":
                out.append(seq[i])
        else:
            if seq[i] != "blank" and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out_seq = "".join(out)
    return out_seq
