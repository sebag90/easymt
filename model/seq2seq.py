import pickle
from pathlib import Path
import random

import torch
import torch.nn as nn


class seq2seq(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            src_lang,
            tgt_lang,
            max_len,
            bpe,
            epoch_trained=0,
            history=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.bpe = bpe
        self.epoch_trained = epoch_trained
        if history is None:
            history = dict()
        self.history = history

    def __repr__(self):
        obj_str = (
            f"Seq2Seq: {self.src_lang.name}-{self.tgt_lang.name} "
            f"[{self.epoch_trained} epoch(s)]\n"
            f"{self.encoder}\n"
            f"{self.decoder}"
        )
        return obj_str

    def save(self, outputpath):
        l1 = self.src_lang.name
        l2 = self.tgt_lang.name
        ep = self.epoch_trained
        path = Path(f"{outputpath}/{l1}-{l2}-ep{ep}.pt")

        with open(path, "wb") as ofile:
            pickle.dump(self, ofile)

    @classmethod
    def load(cls, inputpath):
        with open(inputpath, "rb") as infile:
            obj = pickle.load(infile)
            return obj

    @torch.no_grad()
    def encode(self, batch, lengths, device):
        # move to device and set evaluation mode (dropout)
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

        # prepare input sentence
        batch = batch.to(device)

        # pass through encoder
        encoded = self.encoder(batch, lengths)
        encoder_outputs, encoder_hidden, encoder_cell = encoded

        # create first input for 1st step of decoder
        sos_index = self.src_lang.word2index["SOS"]

        decoder_input = torch.LongTensor(
            [[sos_index for _ in range(batch.shape[1])]]
        )
        context_vector = torch.zeros(
            (batch.shape[1], self.encoder.hidden_size)
        )
        decoder_input = decoder_input.to(device)
        context_vector = context_vector.to(device)

        # rename encoder output for loop decoding
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        return (
            decoder_input, context_vector, decoder_hidden,
            decoder_cell, encoder_outputs
        )

    @torch.no_grad()
    def decoder_step(
            self, decoder_input, context_vector,
            decoder_hidden, decoder_cell, encoder_outputs):
        """
        single step through the decoder
        """
        # pass through decoder
        decoder_output = self.decoder(
            decoder_input,
            context_vector,
            decoder_hidden,
            decoder_cell,
            encoder_outputs
        )
        return decoder_output

    @torch.no_grad()
    def translate_batch(self, batch, lengths, device):
        # move to device and set evaluation mode (dropout)
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

        # prepare input sentence
        batch = batch.to(device)

        # pass through encoder
        encoded = self.encoder(batch, lengths)
        encoder_outputs, encoder_hidden, encoder_cell = encoded

        # create first input for 1st step of decoder
        sos_index = self.src_lang.word2index["SOS"]

        decoder_input = torch.LongTensor(
            [[sos_index for _ in range(batch.shape[1])]],
            device=device
        )
        context_vector = torch.zeros(
            (batch.shape[1], self.encoder.hidden_size),
            device=device
        )

        # rename encoder output for loop decoding
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # empty tensor to store outputs
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)

        # decode word by word -- no teacher
        for i in range(self.max_len):
            # pass through decoder
            (decoder_output, context_vector,
             decoder_hidden, decoder_cell) = self.decoder(
                decoder_input,
                context_vector,
                decoder_hidden,
                decoder_cell,
                encoder_outputs
            )

            # get index of argmax for each element in batch
            _, topi = decoder_output.topk(1)
            # add result to matrix
            all_tokens = torch.cat((all_tokens, topi), dim=1)

            # prepare input for next word prediction
            decoder_input = topi.t()
            decoder_input = decoder_input.to(device)

        return all_tokens

    def train_batch(
            self, batch, device, teacher_forcing_ratio, criterion):
        input_var, lengths, target_var, mask, max_target_len = batch
        len_batch = input_var.shape[1]

        # move batch to device
        input_var = input_var.to(device)
        target_var = target_var.to(device)
        mask = mask.to(device)

        loss = 0

        # pass through encoder
        (encoder_outputs, encoder_hidden,
         encoder_cell) = self.encoder(input_var, lengths)

        # encode input sentence
        sos_index = self.src_lang.word2index["SOS"]
        decoder_input = torch.full(
            (1, len_batch),
            sos_index,
            dtype=torch.int64,
            device=device
        )
        context_vector = torch.zeros(
            (len_batch, self.encoder.hidden_size),
            device=device
        )

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        preds = list()

        for t in range(max_target_len):
            # decide if teacher for next word
            use_teacher_forcing = (
                True if random.random() < teacher_forcing_ratio
                else False
            )

            # pass through decoder
            (decoder_output, context_vector,
             decoder_hidden, decoder_cell) = self.decoder(
                decoder_input,
                context_vector,
                decoder_hidden,
                decoder_cell,
                encoder_outputs
            )

            # collect outputs
            preds.append(decoder_output)

            if use_teacher_forcing:
                # next word is current target
                decoder_input = target_var[t].view(1, -1)
            else:
                # next input is decoder's current output
                _, topi = decoder_output.topk(1)
                decoder_input = topi.t()
                decoder_input = decoder_input.to(device)

        # calculate batch loss
        preds = torch.stack(preds)
        loss = criterion(preds, target_var, mask)
        return loss
