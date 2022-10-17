import numpy as np
import torch
import yaml

from ..models.ocr.CRNN import CRNN
from ..tools.crnn_tools import decode, process_image

with open("./src/configs/paths.yaml") as file:
    models_weights_path = yaml.safe_load(file)["models_weights"]


class CRNNInference:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        crnn_model_weights_path = models_weights_path["path_crnn"]

        self.crnn = CRNN()
        self.crnn.to(self.device)

        self.idx_to_char = dict(zip(range(1, 23), self.crnn.vocabulary))
        self.idx_to_char[0] = "blank"

        with open(crnn_model_weights_path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))

        self.crnn.load_state_dict(state_dict, strict=False)
        self.crnn.to(device)
        self.crnn.eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> str:
        """
        Fulfilling OCR task.

        :param image: cropped car plate image from detection part of size (H, W, C)
        :returns: predicted sequence text or string-warning if prediction is nothing
        """
        img = process_image(image).to(self.device)

        predict = self.crnn(img)
        predict = predict.detach().to("cpu").numpy()
        seq_pred = decode(predict, self.idx_to_char)

        return seq_pred if len(seq_pred) > 0 else "Couldn't recognize any symbols"
