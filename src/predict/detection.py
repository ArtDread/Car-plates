"""The module contains various implementations of car plate detection logic."""
from typing import Dict, List, Optional, TypeAlias

import numpy as np
import torch
import torchvision
import yaml
from numpy.typing import NDArray
from PIL.Image import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..tools.fr_rcnn_tools import apply_nms, preds_but_valid_scores

with open("./src/configs/paths.yaml") as file:
    models_weights_path: Dict[str, str] = yaml.safe_load(file)["models_weights"]

NumpyArray: TypeAlias = NDArray[np.float32]


class FasterRCNNInference:
    """This class is implementation of detection logic with Faster R-CNN model.

    Attributes:
        device: The execution device, cpu or gpu if possible
        fr_rcnn: The PyTorch Faster R-CNN model

    """

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        frrcnn_model_weights_path: str = models_weights_path["path_frrcnn"]
        self.fr_rcnn = FasterRCNNInference.get_model_instance()

        with open(frrcnn_model_weights_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))

        self.fr_rcnn.load_state_dict(state_dict, strict=False)
        self.fr_rcnn.to(self.device)
        self.fr_rcnn.eval()

    @staticmethod
    def get_model_instance(num_classes: int = 2, weights=None):
        """The Faster R-CNN model initialization.

        Args:
            num_classes: The number of predicted classes, i.e. 2 (plate & background)
            weights: The weights configuration for this architecture,
                random initialization by default

        Returns:
            The PyTorch Faster R-CNN model

        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    @torch.no_grad()
    def __call__(self, image: Image) -> Optional[List[NumpyArray]]:
        """Fulfilling the detection task.

        Args:
            image: An image likely including car plate

        Returns:
            List of predicted boxes or None if prediction is very uncertain

        """
        img = torchvision.transforms.ToTensor()(image)

        preds: List[Dict[str, torch.Tensor]] = self.fr_rcnn([img.to(self.device)])
        preds = [
            {k: v.detach().to("cpu") for k, v in prediction.items()}
            for prediction in preds
        ]

        new_preds = preds_but_valid_scores(preds)

        if new_preds:
            return apply_nms(new_preds)
        else:
            return None
