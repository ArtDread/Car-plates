"""The module provides selection of predicted boxes by two criteria."""
from typing import Dict, List, TypeAlias

import numpy as np
import torch
import torchvision
from numpy.typing import NDArray

NumpyArray: TypeAlias = NDArray[np.float32]


def preds_but_valid_scores(
    preds: List[Dict[str, torch.Tensor]], tshd: float = 0.8
) -> List[Dict[str, torch.Tensor]]:
    """Keeps only those predictions for which the score is above the claimed treshold.

    Args:
        preds: An output of the model
        tshd: A threshold of confidence, scalar in range 0-1

    Returns:
        The new_preds containing predicted boxes and scores

    """
    new_preds: List[Dict[str, torch.Tensor]] = []
    for i in range(len(preds)):
        scores = preds[i]["scores"]
        scores = torch.where(scores > tshd, scores, 0.0)
        num = int(scores.count_nonzero().item())
        # If no score greater than thsd then make no predict
        if num != 0:
            preds[i]["boxes"] = preds[i]["boxes"][:num]
            preds[i]["scores"] = preds[i]["scores"][:num]
            del preds[i]["labels"]
            new_preds.append(preds[i])
    return new_preds


def apply_nms(
    preds: List[Dict[str, torch.Tensor]], iou_threshold: float = 0.5
) -> NumpyArray:
    """Apply Non-Maximum Suppression.

    Args:
        preds: An output of the model
        iou_threshold: IoU threshold, scalar in range 0-1

    Returns:
        List of boxes that met condition

    """
    bboxes: List[NumpyArray] = []
    for image in preds:
        inds = torchvision.ops.nms(image["boxes"], image["scores"], iou_threshold)
        image["boxes"] = image["boxes"][inds]
        bboxes.append(image["boxes"].numpy())
    return bboxes[0]
