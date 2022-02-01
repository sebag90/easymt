import torch.nn as nn

from model.transformer.layers import (
    EncoderLayer, DecoderLayer, PositionalEncoding
)


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 dim_ff,
                 attn_dropout,
                 residual_dropout,
                 num_layers,
                 max_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(
            max_len + 1, d_model, residual_dropout
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                n_head,
                dim_ff,
                attn_dropout,
                residual_dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 dim_ff,
                 attn_dropout,
                 residual_dropout,
                 num_layers,
                 vocab_size,
                 max_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(
            max_len + 1, d_model, residual_dropout
        )
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                n_head,
                dim_ff,
                attn_dropout,
                residual_dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)

        return x
