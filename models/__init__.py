import torch
import torch.nn as nn
from deepmeg.models import BaseModel
from layers.bahdanau import BahdanauEncoderDecoderLSTM

class TinyNet(BaseModel):

    def __init__(
        self,
        input_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_size: int,
        n_times: int,
        n_outputs: int
    ):
        super().__init__()
        self.linear = nn.Linear(n_times, n_times//10)
        self.attn_net = BahdanauEncoderDecoderLSTM(
            input_size=input_size,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            attention_size=attention_size
        )
        self.fc_layer = nn.Linear(decoder_hidden_size*n_times//10, n_outputs)


    def forward(self, x: torch.Tensor):
        x = torch.permute(x, (0, 2, 1))
        x = self.linear(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.attn_net(x)
        x = torch.flatten(x, 1)
        return self.fc_layer(x)