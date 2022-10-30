from string import digits
from typing import Tuple

import torch
import torchvision


class FeatureExtractor(torch.nn.Module):
    """This class is implementation of the task-specific CNN model.

    Given the RGB images of size (N, C=3, H, W), create a convolutional
    feature maps, i.e. -> (N, H*, C=512, W*) (see Note for details).

    Note:
        N: The batch size
        C: The number of channels, i.e. const=3 for an RGB image
        H: The height of the image in pixels, i.e. const for batch
        W: The width of the image in pixels, i.e. const for batch
        H*: The new height, it's supposed to be squeezed to 1
        W*: The new width, it could be iterpreted as the number of
            vectors (frames) which is limited to 18 by the russian
                license plate case specifics

    Attributes:
        cnn: The backbone of the PyTorch Resnet-18 model
        pool: The poolying layer for reducing height, i.e. H -> H*
        proj: The convolutional layer for increasing width, i.e. W -> W*
        num_output_features: The feature space dimension

    """

    @property
    def device(self):
        """Moves all model parameters to the device."""
        return next(self.parameters()).device

    def __init__(self, input_size: Tuple[int, int] = (64, 320), output_len: int = 18):
        super(FeatureExtractor, self).__init__()
        h, w = input_size
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        last_basic_block = list(resnet.children())[:-2][-1][-1]

        self.cnn = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.pool = torch.nn.AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = torch.nn.Conv2d(w // 32, output_len, kernel_size=1)
        # Save as input_size for rnn
        self.num_output_features: int = last_basic_block.bn2.num_features

    def _apply_projection(self, x: torch.Tensor):
        x = x.permute(0, 3, 2, 1).contiguous()  # to (N, W, H, C)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # to (N, H, C, W)
        return x

    def forward(self, x: torch.Tensor):
        features = self.cnn(x)  # to (N, C=512, H=2, W=10)
        features = self.pool(features)  # to (N, C=512, H=1, W=10)
        features = self._apply_projection(features)  # to (N, H=1, C=512, W=18)
        return features


class SequencePredictor(torch.nn.Module):
    """This class is implementation of the task-specific RNN model.

    Given the feature sequences of size (N, C=1, H_in, L), make per-frame predictions
    from feature space dimension to vocabulary dimension, i.e. -> (N, L, V)
    (see Note for details).

    Note:
        N: The batch size
        L: The number of task-defined vectors (frames)
        H_in: The feature space dimension, i.e. input_size
        H_out: The size of each hidden state, i.e. hidden_size
        V: The task-defined vocabulary dimension, i.e. num_classes
        D: 2 if bidirectional = True else 1

    Attributes:
        rnn: The PyTorch LSTM model
        fc: The linear layer for transition to vocab dim

    """

    @property
    def device(self):
        """Moves all model parameters to the device."""
        return next(self.parameters()).device

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
    ):
        super(SequencePredictor, self).__init__()

        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        fc_in = hidden_size * int(1 + bidirectional)  # (D*H_out)
        self.fc = torch.nn.Linear(in_features=fc_in, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        x = x.squeeze(dim=1).permute(0, 2, 1)  # to (N, L, H_in)
        output, _ = self.rnn(x)  # to (N, L, D∗H_out)
        output = self.fc(output)  # to (N, L, V)
        return output


class CRNN(torch.nn.Module):
    """This class is implementation of the CRNN model.

    Both implemented CNN and RNN models are applied here.

    Attributes:
        vocabulary: The allowed by car plates task vocabulary
        fts_extr: instance of FeatureExtractor class
        seq_pred: instance of SequencePredictor class

    """

    @property
    def device(self):
        """Moves all model parameters to the device."""
        return next(self.parameters()).device

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        vocabulary: str = digits + "ABEKMHOPCTYX",
        cnn_input_size: Tuple[int, int] = (64, 320),
        output_len: int = 18,
    ):
        super(CRNN, self).__init__()

        self.vocabulary = vocabulary
        self.fts_extr = FeatureExtractor(
            input_size=cnn_input_size, output_len=output_len
        )
        self.seq_pred = SequencePredictor(
            input_size=self.fts_extr.num_output_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=len(vocabulary) + 1,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor):
        features = self.fts_extr(x)
        seq = self.seq_pred(features)
        return seq
