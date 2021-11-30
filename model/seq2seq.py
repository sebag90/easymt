from copy import deepcopy
from pathlib import Path
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lang import Hypothesis


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
            f"Seq2Seq: {self.src_lang.name} --> {self.tgt_lang.name} "
            f"[{self.epoch_trained} epoch(s)]\n"
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
        ep = self.epoch_trained
        path = Path(f"{outputpath}/{l1}-{l2}-ep{ep}.pt")

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

    @torch.no_grad()
    def encode(self, batch, lengths, device):
        """
        encode a batch of sentences for translation
        (batch size = 1)
        """
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

    def train_batch(
            self, batch, device, teacher_forcing_ratio, criterion):
        """
        calculate and return the error on a mini batch
        """
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

            step_loss = criterion(
                decoder_output, target_var[t], mask[t]
            )
            loss += step_loss

            if use_teacher_forcing:
                # next word is current target
                decoder_input = target_var[t].view(1, -1)
            else:
                # next input is decoder's current output
                _, topi = decoder_output.topk(1)
                decoder_input = topi.t()
                decoder_input = decoder_input.to(device)

        # calculate batch loss
        return loss / max_target_len

    @torch.no_grad()
    def beam_search(self, line, beam_size, device):
        """
        beam translation for a single line of text
        """
        # encode line
        coded = self.src_lang.indexes_from_sentence(line.strip())
        lengths = torch.tensor([len(coded)])
        input_batch = coded.unsqueeze(1)

        # encode sentence
        (decoder_input, *decoder_state, encoder_outputs) = self.encode(
            input_batch, lengths, device
        )

        complete_hypotheses = list()
        live_hypotheses = list()

        # create empty hypothesis with only SOS token
        hyp = Hypothesis()
        hyp.update(
            decoder_input, decoder_state, 0
        )
        live_hypotheses.append(hyp)

        # begin beam search
        t = 0
        while t < self.max_len and len(complete_hypotheses) < beam_size:
            step_hypotheses = list()

            # iterate through hypotheses and explore them
            for hypothesis in live_hypotheses:
                # obtain last decoder input from hypothesis
                decoder_input = hypothesis.last_word
                decoder_state = hypothesis.decoder_state

                # pass through the decoder
                decoder_output, *decoder_state = self.decoder_step(
                    decoder_input,
                    *decoder_state,
                    encoder_outputs
                )

                # pass decoder output through softmax
                # to obtain negative log likelihood
                decoder_output = F.log_softmax(decoder_output, dim=-1)

                # obtain k (number of alive threads)
                k = beam_size - len(complete_hypotheses)

                # obtain k best options
                probs, indeces = decoder_output.topk(k)
                indeces = indeces.squeeze()
                probs = probs.squeeze()

                # adjust dimensions if k == 1
                if len(indeces.shape) == 0:
                    indeces = indeces.unsqueeze(0)
                    probs = probs.unsqueeze(0)

                # iterate through the k most possible
                for i in range(k):
                    # get decoded index and its log prob
                    decoded = indeces[i]
                    log_prob = probs[i].item()

                    # create new hypothesis based on current one
                    new_hyp = deepcopy(hypothesis)

                    # update last word, score and decoder_state
                    token_id = decoded.unsqueeze(0).unsqueeze(0)

                    new_hyp.update(
                        token_id, decoder_state, log_prob
                    )

                    # complete hypothesis if decoded EOS
                    idx = decoded.item()
                    if self.tgt_lang.index2word[idx] == "EOS":
                        complete_hypotheses.append(new_hyp)
                    else:
                        step_hypotheses.append(new_hyp)

            # prune the k best live_hypotheses
            step_hypotheses.sort(reverse=True)
            live_hypotheses = step_hypotheses[:k]

            # increase t
            t += 1

        # if no complete hypothesis, use the alive ones
        if len(complete_hypotheses) == 0:
            complete_hypotheses = live_hypotheses

        # pick most probabile hypothesis
        complete_hypotheses.sort(reverse=True)

        # return sorted hypotheses
        return complete_hypotheses
