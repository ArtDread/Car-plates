from string import digits

import torch
import torchvision


class FeatureExtractor(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, input_size=(64, 320), output_len=18):
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
        self.num_output_features = last_basic_block.bn2.num_features

    def _apply_projection(self, x):
        # (N, C, H, W) to (N, W, H, C)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        # Return dimensions order back
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x):
        features = self.cnn(x)
        features = self.pool(features)
        features = self._apply_projection(features)
        return features


class SequencePredictor(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, dropout, bidirectional
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

        fc_in = hidden_size * (1 + int(bidirectional))
        self.fc = torch.nn.Linear(in_features=fc_in, out_features=num_classes)

    def forward(self, x):
        # Make permutation
        x = x.squeeze(dim=1).permute(0, 2, 1)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


class CRNN(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        vocabulary=digits + "ABEKMHOPCTYX",
        cnn_input_size=(64, 320),
        output_len=18,
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

    def forward(self, x):
        features = self.fts_extr(x)
        seq = self.seq_pred(features)
        return seq
