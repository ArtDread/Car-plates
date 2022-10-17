from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision
import yaml
from PIL.Image import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

with open("./src/configs/paths.yaml") as file:
    models_weights_path = yaml.safe_load(file)["models_weights"]


class FasterRCNNInference:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        frrcnn_model_weights_path = models_weights_path["path_frrcnn"]
        self.fr_rcnn = FasterRCNNInference.get_model_instance()

        with open(frrcnn_model_weights_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))

        self.fr_rcnn.load_state_dict(state_dict, strict=False)
        self.fr_rcnn.to(self.device)
        self.fr_rcnn.eval()

    @staticmethod
    def get_model_instance(num_classes: int = 2, weights=None):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _preds_but_valid_scores(
        self, preds: List[Dict[str, torch.Tensor]], tshd: float = 0.8
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Keeps only those predictions for which the score is above the claimed treshold.

        Args:
            param preds: output of the model
            param tshd: threshold of confidence, scalar in range 0-1

        Returns:
            preds
        """
        new_preds = []
        for i in range(len(preds)):
            scores = preds[i]["scores"]
            scores = torch.where(scores > tshd, scores, 0.0)
            num = scores.count_nonzero().item()
            # If no score greater than thsd then make no predict
            if num != 0:
                preds[i]["boxes"] = preds[i]["boxes"][:num]
                preds[i]["scores"] = preds[i]["scores"][:num]
                del preds[i]["labels"]
                new_preds.append(preds[i])
        return new_preds

    def _apply_nms(
        self, preds: List[Dict[str, torch.Tensor]], iou_threshold: float = 0.5
    ) -> List[np.ndarray]:
        """
        Apply Non-Maximum Suppression.

        Args:
            param preds: output of the model
            param iou_threshold: iou threshold, scalar in range 0-1

        Returns:
            List of boxes that met condition
        """
        bboxes = []
        for image in preds:
            inds = torchvision.ops.nms(image["boxes"], image["scores"], iou_threshold)
            image["boxes"] = image["boxes"][inds]
            bboxes.append(image["boxes"].numpy())
        return bboxes

    @torch.no_grad()
    def __call__(self, image: Image) -> Optional[List[np.ndarray]]:
        """
        Fulfilling the Detection task.

        Args:
            param image: image likely including car plate

        Returns:
            List of predicted boxes or None if prediction is very uncertain
        """
        img = torchvision.transforms.ToTensor()(image)

        preds = self.fr_rcnn([img.to(self.device)])
        preds = [
            {k: v.detach().cpu() for k, v in prediction.items()} for prediction in preds
        ]

        preds = self._preds_but_valid_scores(preds)

        if preds:
            return self._apply_nms(preds)
        else:
            return None
