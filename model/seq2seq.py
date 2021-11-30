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

        # decoder_state:
        # 1 - context vector
        # 2 - decoder_hidden
        # 3 - decoder_cell
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

            # collect tensors from live hypotheses
            # to create input batch (batch size = beam size)
            step_input = list()
            step_context_vector = list()
            step_hidden = list()
            step_cell = list()
            step_encoder_outputs = list()

            for hypothesis in live_hypotheses:
                step_input.append(hypothesis.last_word)
                step_encoder_outputs.append(encoder_outputs)
                (context_vector,
                 decoder_hidden,
                 decoder_cell) = hypothesis.decoder_state
                step_context_vector.append(context_vector)
                step_hidden.append(decoder_hidden)
                step_cell.append(decoder_cell)

            # create batch with live hypotheses
            decoder_input = torch.cat(step_input, dim=1)
            context_vector = torch.cat(step_context_vector)
            decoder_hidden = torch.cat(step_hidden, dim=1)
            decoder_cell = torch.cat(step_cell, dim=1)
            enc_outputs = torch.cat(step_encoder_outputs, dim=1)

            # pass through decoder
            decoder_output, attention_output, hidden, cell = self.decoder(
                decoder_input,
                context_vector,
                decoder_hidden,
                decoder_cell,
                enc_outputs
            )

            # softmax to get negative log likelihood to sum scores
            decoder_output = F.log_softmax(decoder_output, dim=-1)

            # decide k and get best options
            k = beam_size - len(complete_hypotheses)
            probs, indeces = decoder_output.topk(k)

            # for each live hypothesis explore k alternatives
            for i, k_bests in enumerate(zip(probs, indeces)):
                best_probs, best_indeces = k_bests
                hypothesis = live_hypotheses[i]

                # save attention, hidden and cell tensors
                this_context = attention_output[i].unsqueeze(0)
                this_hidden = hidden[:, i, :].unsqueeze(1)
                this_cell = cell[:, i, :].unsqueeze(1)

                decoder_state = (
                    this_context, this_hidden, this_cell
                )

                for log_prob, decoded in zip(best_probs, best_indeces):
                    # create a new hypothesis for each k alternative
                    new_hyp = deepcopy(hypothesis)
                    token_id = decoded.unsqueeze(0).unsqueeze(0)

                    # update hypothesis with new word and score
                    new_hyp.update(
                        token_id, decoder_state, log_prob.item()
                    )

                    # complete hypothesis if decoded EOS
                    idx = decoded.item()
                    if self.tgt_lang.index2word[idx] == "EOS":
                        complete_hypotheses.append(new_hyp)
                    else:
                        step_hypotheses.append(new_hyp)

            # update k for pruning
            k = beam_size - len(complete_hypotheses)

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
