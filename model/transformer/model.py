from pathlib import Path
import pickle

import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            src_lang,
            tgt_lang,
            max_len):
        super().__init__()
        self.type = "transformer"
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
            f"Transformer({self.src_lang.name} > {self.tgt_lang.name} | "
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
        path = Path(f"{outputpath}/{self.type}_{l1}-{l2}_{st}.pt")

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

    def train_batch(self, batch, device, teacher_forcing_ratio, criterion):
        input_var, decoder_input, target_var, e_mask, d_mask = batch

        input_var.to(device)
        decoder_input.to(device)
        target_var.to(device)
        e_mask.to(device)
        d_mask.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        encoded = self.encoder(input_var, e_mask)
        decoded = self.decoder(decoder_input, encoded, e_mask, d_mask)
        loss = criterion(decoded.view(-1, decoded.size(-1)), target_var.view(-1))
        return loss