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
            max_len, d_model, residual_dropout
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

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


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
            max_len, d_model, residual_dropout
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

        self.norm = nn.LayerNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)

        x = self.norm(x)
        output = self.generator(x)
        return output
