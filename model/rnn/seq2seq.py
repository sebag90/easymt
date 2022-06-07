from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lang import Hypothesis


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class seq2seq(nn.Module):
    def __init__(
            self, encoder, decoder, embedding, src_lang, tgt_lang, max_len):
        super().__init__()
        self.type = "rnn"
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.steps = 0
        self.size = encoder.hidden_size

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
            f"{self.embedding}\n"
            f"{self.encoder}\n"
            f"{self.decoder}"
        )
        return obj_str

    def prepare_batch(self, src, tgt):
        src = [self.src_lang.toks2idx(sen) for sen in src]
        tgt = [self.tgt_lang.toks2idx(sen) for sen in tgt]

        # prepare source data
        src_len = torch.tensor([len(indexes) for indexes in src])
        src_pad = nn.utils.rnn.pad_sequence(src)

        # prepare target data
        max_tgt_len = max([len(indexes) for indexes in tgt])
        tgt_pad = nn.utils.rnn.pad_sequence(tgt)

        # prepare mask
        mask = tgt_pad != 0

        return src_pad, src_len, tgt_pad, mask, max_tgt_len

    def encode(self, input_var, lengths):
        """
        encode a batch of sentences for translation
        (batch size = 1)
        """
        # prepare input sentence
        input_var = input_var.to(DEVICE)
        len_batch = input_var.shape[1]

        # obtain embedded input sequence
        embedded = self.embedding.src(input_var)

        # pass through encoder
        (encoder_outputs,
         encoder_hidden,
         encoder_cell) = self.encoder(embedded, lengths)

        # prepare decoder input
        sos_index = self.src_lang.word2index["<sos>"]
        decoder_input = torch.full(
            (1, len_batch),
            sos_index,
            dtype=torch.int,
            device=DEVICE
        )

        context_vector = torch.zeros(
            (len_batch, self.encoder.hidden_size),
            device=DEVICE
        )

        return (
            decoder_input, context_vector, encoder_hidden,
            encoder_cell, encoder_outputs
        )

    def forward(self, batch, teacher_forcing_ratio, criterion):
        """
        calculate and return the error on a mini batch
        """
        (input_var,
         lengths,
         target_var,
         mask,
         max_target_len) = self.prepare_batch(*batch)

        # move batch to device
        input_var = input_var.to(DEVICE)
        target_var = target_var.to(DEVICE)
        mask = mask.to(DEVICE)

        loss = 0

        # encode input sentence
        (decoder_input,
         context_vector,
         decoder_hidden,
         decoder_cell,
         encoder_outputs) = self.encode(input_var, lengths)

        for t in range(max_target_len):
            # decide if teacher for next word
            use_teacher_forcing = (
                True if random.random() < teacher_forcing_ratio
                else False
            )

            decoder_input = self.embedding.tgt(decoder_input)

            # pass through decoder
            (decoder_output, context_vector,
             decoder_hidden, decoder_cell) = self.decoder(
                decoder_input,
                context_vector,
                decoder_hidden,
                decoder_cell,
                encoder_outputs
            )

            # compute loss
            step_loss = criterion(
                decoder_output, target_var[t]
            )
            loss += step_loss

            # choose next word
            if use_teacher_forcing:
                # next word is current target
                decoder_input = target_var[t].view(1, -1)
            else:
                # next input is decoder's current output
                _, topi = decoder_output.topk(1)
                decoder_input = topi.t()
                decoder_input = decoder_input.to(DEVICE)

        # calculate batch loss
        return loss / max_target_len

    @torch.no_grad()
    def beam_search(self, line, beam_size, alpha):
        """
        beam translation for a single line of text
        """
        # encode line
        coded = self.src_lang.toks2idx(line.strip().split())
        lengths = torch.tensor([len(coded)])
        input_batch = coded.unsqueeze(1)

        # decoder_state:
        # 1 - context vector
        # 2 - decoder_hidden
        # 3 - decoder_cell
        (decoder_input, *decoder_state, encoder_outputs) = self.encode(
            input_batch, lengths
        )

        complete_hypotheses = list()
        live_hypotheses = list()

        # create empty hypothesis with only <sos> token
        hyp = Hypothesis(alpha=alpha)
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

            # obtain input words and decoder state
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

            decoder_input = self.embedding.tgt(decoder_input)

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

                for log_prob, index in zip(best_probs, best_indeces):
                    # create a new hypothesis for each k alternative
                    new_hyp = deepcopy(hypothesis)
                    token_id = index.unsqueeze(0).unsqueeze(0)

                    # update hypothesis with new word and score
                    new_hyp.update(
                        token_id, decoder_state, log_prob.item()
                    )

                    # complete hypothesis if decoded <eos>
                    idx = index.item()
                    if self.tgt_lang.index2word[idx] == "<eos>":
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
