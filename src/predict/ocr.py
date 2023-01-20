"""The module contains various implementations of a car plate recognition logic."""
from string import digits
from typing import Dict, List, Tuple, TypeAlias

import easyocr  # type:ignore
import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from PIL.Image import Image

from ..models.ocr.CRNN import CRNN
from ..tools.crnn_tools import decode, process_image

with open("./src/configs/paths.yaml") as file:
    models_weights_path: Dict[str, str] = yaml.safe_load(file)["models_weights"]

NumpyArray: TypeAlias = NDArray[np.float32]
EasyOCROutcome: TypeAlias = List[Tuple[List[List[int]], str, float]]


class CRNNInference:
    """This class is implementation of a recognition logic with CRNN model.

    Attributes:
        device: The execution device, cpu or gpu if possible
        crnn: The PyTorch CRNN model
        idx_to_char: The vocabulary to return to allowed plates' chars while decoding

    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        crnn_model_weights_path: str = models_weights_path["path_crnn"]

        self.crnn = CRNN()
        self.crnn.to(self.device)

        self.idx_to_char: Dict[int, str] = dict(zip(range(1, 23), self.crnn.vocabulary))
        self.idx_to_char[0] = "blank"

        with open(crnn_model_weights_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))

        self.crnn.load_state_dict(state_dict, strict=False)
        self.crnn.to(device)
        self.crnn.eval()

    @torch.no_grad()
    def __call__(self, image: Image) -> str:
        """Fulfilling the OCR task.

        Args:
            image: The cropped car plate image from detection part

        Returns:
            Predicted sequence text or string-warning if prediction is nothing

        """
        img = process_image(image).to(self.device)

        predict: torch.Tensor = self.crnn(img)
        pred_seq: NumpyArray = predict.to("cpu").numpy()
        seq_pred = decode(pred_seq, self.idx_to_char)

        return seq_pred if len(seq_pred) > 0 else "Couldn't recognize any symbols"


class EasyOCRInference:
    """This class is implementation of a recognition logic with easyOCR framework.

    Attributes:
        device: The execution device, cpu or gpu if possible
        easy_default: The EasyOCR Reader model
        characters_subset: Force EasyOCR to recognize only specific subset of characters

    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.easy_default = easyocr.Reader(["en"], gpu=(self.device.type != "cpu"))
        self.characters_subset = digits + "ABEKMHOPCTYX"

    def __call__(self, image: Image) -> str:
        """Fulfilling the OCR task.

        Args:
            image: The cropped car plate image from detection part

        Returns:
            Predicted sequence text or string-warning if prediction is nothing

        """
        image_numpy = np.asarray(image).astype(np.uint8)  # to shape (H, W, C)
        predictions: EasyOCROutcome = self.easy_default.readtext(
            image_numpy, allowlist=self.characters_subset
        )

        # Remove spaces
        processed_seqs = ["".join(item[-2].split()) for item in predictions]
        # Merge sequences if a few
        seq_pred = "".join(processed_seqs)

        return seq_pred if len(seq_pred) > 0 else "Couldn't recognize any symbols"
