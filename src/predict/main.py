import sys
import time
from typing import Dict, List, Literal, Optional, Tuple, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image as Img
from PIL import ImageDraw as ImgDraw
from PIL import ImageFont as ImgFont
from PIL.Image import Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont

from ..tools.find_font_size import find_font_size
from .detection import CascadeClassifierInference, FasterRCNNInference
from .ocr import CRNNInference, EasyOCRInference

Outcome: TypeAlias = Optional[
    Tuple[List[str], Optional[Image], Optional[Dict[str, float]]]
]

DetectionModel: TypeAlias = Literal["fasterrcnn", "cascadeclassifier"]
DetectionModelAttr: TypeAlias = Union[FasterRCNNInference, CascadeClassifierInference]
RecognitionModelAttr: TypeAlias = Union[CRNNInference, EasyOCRInference]
OcrModel: TypeAlias = Literal["crnn", "easyocr"]
NumpyArray: TypeAlias = NDArray[np.float32]


class GeneralInference:
    """This class is implementation of pipeline logic.

    A prediction of a license plate sequence from end-to-end
    by combining both car plate detection & OCR parts.

    Attributes:
        detection_model: The model selected for detection
        ocr_model: The model selected for OCR
        display: Display box and license plate sequence on output or not
        debug: Save runtime value for each part or not
        font: A path to filename containing a TrueType font
        combination_name: The name of chosen model combination

    """

    def __init__(
        self,
        detection_model: DetectionModel = "fasterrcnn",
        ocr_model: OcrModel = "crnn",
        display: bool = True,
        debug: bool = False,
        font: str = "./src/tools/fonts/Roboto-Medium.ttf",
    ):
        self.detection_model: DetectionModelAttr
        if detection_model == "fasterrcnn":
            self.detection_model = FasterRCNNInference()
        elif detection_model == "cascadeclassifier":
            self.detection_model = CascadeClassifierInference()
        else:
            raise ValueError("The given detection model not found")

        self.ocr_model: RecognitionModelAttr
        if ocr_model == "crnn":
            self.ocr_model = CRNNInference()
        elif ocr_model == "easyocr":
            self.ocr_model = EasyOCRInference()
        else:
            raise ValueError("The given ocr model not found")

        self.display = display
        self.debug = debug
        self.font = font
        self.combination_name = f"{detection_model} + {ocr_model}"

    def _crop_carplate_image(self, image: Image, bbox: NumpyArray) -> Image:
        return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    def _demonstration(self, image: Image, bbox: NumpyArray, text: str) -> Image:
        font_size = find_font_size(text, self.font, image, 0.08)
        roboto_font: ImageFont = ImgFont.truetype(self.font, font_size)
        draw: ImageDraw = ImgDraw.Draw(image)

        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline=128, width=4)
        draw.text(
            (bbox[0], bbox[3]),
            text,
            fill=(255, 128, 0),
            font=roboto_font,
            stroke_width=round(0.05 * font_size),
            stroke_fill=(0, 0, 0),
        )
        return image

    def detect_by_image(self, image: Image) -> Outcome:
        """
        Run loaded image through parts of pipeline and get the predict.

        Args:
            image: A car picture, probably containing a distinguishable license plate

        Returns:
            A list of predicted car plates (or None), an image display with drawn
                boundary boxes and predicted car plates labels (or None),
                    detection & OCR runtime (or None)

        """
        ocr_results: List[str] = []
        display_image: Optional[Image] = None
        if self.display:
            display_image = image.copy()

        start_time_det: float = time.time()
        detection_results: Optional[NumpyArray] = self.detection_model(image)
        end_time_det: float = time.time()

        start_time_ocr: float = time.time()
        if detection_results is not None:
            for bbox in detection_results:
                img = self._crop_carplate_image(image, bbox)

                seq_recognition: str = self.ocr_model(img)
                ocr_results.append(seq_recognition)
                if self.display:
                    display_image = self._demonstration(
                        display_image, bbox, seq_recognition
                    )
        end_time_ocr: float = time.time()

        if self.debug:
            debug_info: Dict[str, float] = {
                "detection_time": end_time_det - start_time_det,
                "ocr_time": end_time_ocr - start_time_ocr,
            }
            return ocr_results, display_image, debug_info
        else:
            return ocr_results, display_image, None

    def detect_by_image_path(self, path_to_image: str) -> Outcome:
        try:
            image: Image = Img.open(path_to_image)
            return self.detect_by_image(image)
        except OSError as e:
            print(f"Unable to open {path_to_image}: {e}", file=sys.stderr)
            return None
