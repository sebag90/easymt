import torch.nn as nn

from model.transformer_modules import (
    EncoderLayer, DecoderLayer, PositionalEncoding, LayerNormalizer
)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 n_embed,
                 n_head,
                 dim_ff,
                 attn_dropout,
                 residual_dropout,
                 num_layers,
                 vocab_size,
                 word_vec_size,
                 max_len):
        super().__init()
        self.embedding = nn.Embedding(
            vocab_size,
            word_vec_size,
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

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self,
                 n_embed,
                 n_head,
                 dim_ff,
                 attn_dropout,
                 residual_dropout,
                 num_layers,
                 vocab_size,
                 word_vec_size,
                 max_len):
        super().__init()
        self.embedding = nn.Embedding(
            vocab_size,
            word_vec_size,
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


class Transformer(nn.Module):
    def __init__(self,
                 n_embed,
                 n_head,
                 dim_ff,
                 attn_dropout,
                 residual_dropout,
                 encoder_layers,
                 decoder_layers,
                 vocab_size,
                 word_vec_size,
                 max_len):
        super().__init()
        self.encoder = TransformerEncoder(
            n_embed,
            n_head,
            dim_ff,
            attn_dropout,
            residual_dropout,
            encoder_layers,
            vocab_size,
            word_vec_size,
            max_len
        )
        self.decoder = TransformerDecoder(
            n_embed,
            n_head,
            dim_ff,
            attn_dropout,
            residual_dropout,
            decoder_layers,
            vocab_size,
            word_vec_size,
            max_len
        )
