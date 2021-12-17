from pathlib import Path
import pickle

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
        super().__init__()
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

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

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
        super().__init__()
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
    def __init__(
            self,
            encoder,
            decoder,
            src_lang,
            tgt_lang,
            max_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.steps = 0

    def __repr__(self):
        # count trainable parameters
        parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        # create print string
        obj_str = (
            f"Seq2Seq({self.src_lang.name} > {self.tgt_lang.name} | "
            f"steps: {self.steps:,} | "
            f"parameters: {parameters:,})\n"
            f"{self.encoder}\n"
            f"{self.decoder}"
        )
        return obj_str

    def save(self, outputpath):
        """
        save model to a pickle file
        """
        l1 = self.src_lang.name
        l2 = self.tgt_lang.name
        st = self.steps
        path = Path(f"{outputpath}/{l1}-{l2}_{st}.pt")

        with open(path, "wb") as ofile:
            pickle.dump(self, ofile)

    @classmethod
    def load(cls, inputpath):
        """
        load model from pickle file
        """
        with open(inputpath, "rb") as infile:
            obj = pickle.load(infile)
            return obj


if __name__ == "__main__":
    from utils.lang import Language
    from utils.dataset import BatchedData
    
    encoder = TransformerEncoder(256, 4, 512, 0.2, 0.1, 3, 40000, 256, 200)
    decoder = TransformerDecoder(256, 4, 512, 0.2, 0.1, 2, 40000, 256, 200)
    l1 = Language("en")
    l2 = Language("it")

    data = BatchedData(Path("data/batched"))

    transformer = Transformer(encoder, decoder, l1, l2, 200)
    
    for batch in data:
        input_var, lengths, target_var, mask, max_target_len = batch
        input_var = input_var.transpose(1, 0)
        target_var = target_var.transpose(1, 0)
        import torch

        input_var = torch.nn.ZeroPad2d((-1, 201 - input_var.shape[1]))(input_var)
        target_var = torch.nn.ZeroPad2d((-1, 201 - target_var.shape[1]))(target_var)

        e_mask = (input_var != 0).unsqueeze(1)
        encoded = encoder(input_var, e_mask)
        breakpoint()