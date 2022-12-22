"""The module provides processing of image and logic for decoding CRNN's output."""
from typing import Dict, List, TypeAlias

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL.Image import Image

NumpyArray: TypeAlias = NDArray[np.float32]


def process_image(image: Image) -> torch.Tensor:
    """Transform the image format into tensor from Image.

    Args:
        image: The image of cropped car plate, size (H, W, C)

    Returns:
        The transformed image, size (1, C, H, W)

    """
    image_numpy = np.asarray(image).astype(np.float32) / 255.0
    image_numpy = cv2.resize(image_numpy, (320, 64), interpolation=cv2.INTER_AREA)
    image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).float().unsqueeze(0)
    return image_tensor


def decode(pred_seq: NumpyArray, idx_to_char: Dict[int, str]) -> str:
    """Convert predicted indexed tokens by crnn model to chars.

    Args:
        pred_seq: CRNN's output, size (N=1, L, V),
            where L - the number of task-defined vectors (frames),
                V - the task-defined vocabulary dimension
        idx_to_char: Vocabulary for decoding indexed tokens

    Returns:
        The decoded car plate text sequence

    """
    pred_seq = pred_seq[0]
    seq: List[str] = []  # Consists of blanks
    for i in range(len(pred_seq)):
        # Define predicted class in indexed form
        label_idx = np.argmax(pred_seq[i]).item()
        # Convert to char
        seq.append(idx_to_char[label_idx])

    out: List[str] = []  # Consists of no blanks
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != "blank":
                out.append(seq[i])
        else:
            if seq[i] != "blank" and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out_seq = "".join(out)
    return out_seq
