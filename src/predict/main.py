import sys
import time
from typing import Dict, List, Optional, Tuple, TypeAlias

# import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image as Img
from PIL import ImageDraw, ImageFont
from PIL.Image import Image

from .detection import FasterRCNNInference
from .ocr import CRNNInference

Outcome: TypeAlias = Optional[
    Tuple[List[str], Optional[Image], Optional[Dict[str, float]]]
]


class GeneralInference:
    def __init__(
        self,
        detection_model: str = "fasterrcnn",
        ocr_model: str = "crnn",
        display: bool = True,
        debug: bool = False,
        font_path: str = "./src/utils/fonts/Roboto-Medium.ttf",
    ):
        """
        detection_models (str): fasterrcnn /  ...
        ocr_models (str): crnn / ...
        """
        if detection_model == "fasterrcnn":
            self.detection_model = FasterRCNNInference()
        else:
            raise ValueError("Given detection model not found")

        if ocr_model == "crnn":
            self.ocr_model = CRNNInference()
        else:
            raise ValueError("Given ocr model not found")

        self.display = display
        self.debug = debug
        self.font_path = font_path

    def _crop_carplate_image(self, image: Image, bbox: NDArray[np.float32]) -> Image:
        return image.crop((bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]))

    def _demonstration(
        self, image: Image, bbox: NDArray[np.float32], text: str
    ) -> Image:
        # TODO: fontsize proportional to image size
        roboto_font = ImageFont.truetype(self.font_path, 40)
        draw = ImageDraw.Draw(image)
        # Draw rectangle
        draw.rectangle(
            (bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]), outline=128, width=4
        )
        draw.text((bbox[0][0], bbox[0][3]), text, fill=(255, 128, 0), font=roboto_font)
        return image

    def detect_by_image(self, image: Image) -> Outcome:
        """
        Load .

        Args:
            image: a picture of a car, probably containing a lisense plate

        Returns:
            Image with drawn bboxes having predicted car plates labels
        """
        ocr_results = []
        display_image = None
        if self.display:
            display_image = image.copy()

        start_time_det = time.time()
        detection_results = self.detection_model(image)
        end_time_det = time.time()

        start_time_ocr = time.time()
        if detection_results is not None:
            for bbox in detection_results:
                image = (
                    np.asarray(self._crop_carplate_image(image, bbox)).astype(
                        np.float32
                    )
                    / 255.0
                )

                seq_recognition = self.ocr_model(image)
                ocr_results.append(seq_recognition)
                if self.display:
                    display_image = self._demonstration(
                        display_image, bbox, seq_recognition
                    )
        end_time_ocr = time.time()

        if self.debug:
            debug_info = {
                "detection_time": end_time_det - start_time_det,
                "ocr_time": end_time_ocr - start_time_ocr,
            }
            return ocr_results, display_image, debug_info
        else:
            return ocr_results, display_image, None

    def detect_by_image_path(self, path_to_image: str) -> Outcome:
        """
        Load image by path and call detect_by_image if succseed.

        Args:
            param preds: path to image

        Returns:
            detect_by_image()

        Raises:
            OSError: if path is not valid
        """
        try:
            image = Img.open(path_to_image)
            return self.detect_by_image(image)
        except OSError as e:
            print(f"Unable to open {path_to_image}: {e}", file=sys.stderr)
            return None
