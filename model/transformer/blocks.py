import torch.nn as nn

from model.transformer.layers import (
    EncoderLayer, DecoderLayer, PositionalEncoding, LayerNormalizer
)


class Encoder(nn.Module):
    def __init__(self,
                 n_embed,
                 n_head,
                 dim_ff,
                 attn_dropout,
                 residual_dropout,
                 num_layers,
                 vocab_size,
                 max_len):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            n_embed,
            padding_idx=0
        )
        self.positional_encoding = PositionalEncoding(
            max_len, n_embed, residual_dropout
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                n_embed,
                n_head,
                dim_ff,
                attn_dropout,
                residual_dropout
            ) for _ in range(num_layers)
        ])

        self.norm = LayerNormalizer(n_embed)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self,
                 n_embed,
                 n_head,
                 dim_ff,
                 attn_dropout,
                 residual_dropout,
                 num_layers,
                 vocab_size,
                 max_len):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            n_embed,
            padding_idx=0
        )
        self.positional_encoding = PositionalEncoding(
            max_len, n_embed, residual_dropout
        )
        self.layers = nn.ModuleList([
            DecoderLayer(
                n_embed,
                n_head,
                dim_ff,
                attn_dropout,
                residual_dropout
            ) for _ in range(num_layers)
        ])

        self.norm = LayerNormalizer(n_embed)
        self.out = nn.Linear(n_embed, vocab_size)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)

        x = self.norm(x)
        output = self.out(x)
        return output
