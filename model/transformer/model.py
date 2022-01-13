from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lang import Hypothesis


class Transformer(nn.Module):
    def __init__(
            self, encoder, decoder, embedding, src_lang, tgt_lang, max_len):
        super().__init__()
        self.type = "transformer"
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.steps = 0
        self.size = encoder.d_model

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
            f"{self.embedding}\n"
            f"{self.encoder}\n"
            f"{self.decoder}"
        )
        return obj_str

    def prepare_batch(self, src, tgt):
        # create target variables by removing <eos> token
        decoder_input = list()
        for sentence in tgt:
            decoder_input.append(sentence[:-1])

        decoder_input = nn.utils.rnn.pad_sequence(
            decoder_input, batch_first=True
        )
        src = nn.utils.rnn.pad_sequence(
            src, batch_first=True
        )
        target = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True
        )

        # add <sos> padding to decoder input
        sos = self.tgt_lang.word2index["<sos>"]
        sos_padder = torch.nn.ConstantPad2d((1, 0, 0, 0), sos)
        decoder_input = sos_padder(decoder_input)

        # create masks
        e_mask = (src != 0).unsqueeze(1)
        d_mask = (decoder_input != 0).unsqueeze(1)
        subseq_mask = torch.ones(
            (1, decoder_input.size(1), decoder_input.size(1)),
            dtype=torch.bool
        )
        subseq_mask = torch.tril(subseq_mask)
        d_mask = torch.logical_and(d_mask, subseq_mask)

        return src, decoder_input, target, e_mask, subseq_mask

    def forward(self, batch, device, teacher_forcing_ratio, criterion):
        input_var, decoder_input, target_var, e_mask, d_mask = batch

        # move tensors to device
        input_var = input_var.to(device)
        decoder_input = decoder_input.to(device)
        target_var = target_var.to(device)
        e_mask = e_mask.to(device)
        d_mask = d_mask.to(device)

        # obtain embedded sequences
        input_var = self.embedding.src(input_var)
        decoder_input = self.embedding.tgt(decoder_input)

        # pass through encoder and decoder
        encoded = self.encoder(input_var, e_mask)
        decoded = self.decoder(decoder_input, encoded, e_mask, d_mask)

        # calculate and return loss
        loss = criterion(
            decoded.view(-1, decoded.size(-1)),
            target_var.view(-1)
        )
        return loss

    @staticmethod
    def create_subsequent_mask(size):
        subseq_mask = torch.ones((1, size, size), dtype=torch.bool)
        subseq_mask = torch.tril(subseq_mask)
        return subseq_mask

    @torch.no_grad()
    def beam_search(self, line, beam_size, device, alpha):
        """
        beam translation for a single line of text
        """
        # encode line
        coded = self.src_lang.toks2idx(line.strip().split())
        padder = torch.nn.ZeroPad2d((0, self.max_len - coded.size(0)))
        src = padder(coded).unsqueeze(0)
        e_mask = (src != 0)

        src = src.to(device)
        e_mask = e_mask.to(device)

        src = self.embedding.src(src)
        encoded = self.encoder(src, e_mask)

        # prepare decoder input
        sos_index = self.src_lang.word2index["<sos>"]
        d_mask = self.create_subsequent_mask(self.max_len)

        complete_hypotheses = list()
        live_hypotheses = list()

        # create empty hypothesis with only <sos> token
        hyp = Hypothesis(alpha=alpha)
        hyp.update(
            torch.tensor([sos_index]), torch.zeros(1), 0
        )
        live_hypotheses.append(hyp)

        # begin beam search
        t = 0
        while t < self.max_len and len(complete_hypotheses) < beam_size:
            step_hypotheses = list()

            # collect tensors from live hypotheses
            # to create input batch (batch size = beam size)
            step_input = list()
            encoder_output = list()

            # obtain input words and encoder_output
            for hypothesis in live_hypotheses:
                step_input.append(torch.tensor(hypothesis.tokens))
                encoder_output.append(encoded)

            # create batch with live hypotheses
            decoder_input = torch.vstack(step_input).to(device)
            encoder_output = torch.cat(encoder_output)

            # create mask for decoding
            d_mask = self.create_subsequent_mask(decoder_input.size(1)).to(device)

            # pass through decoder
            decoder_input = self.embedding.tgt(decoder_input)
            decoded = self.decoder(
                decoder_input, encoder_output, e_mask, d_mask
            )

            # softmax to get negative log likelihood to sum scores
            decoder_output = F.log_softmax(decoded, dim=-1)

            # only get the last word for each beam
            decoder_output = decoder_output[:, -1, :]

            # decide k and get best options
            k = beam_size - len(complete_hypotheses)
            probs, indeces = decoder_output.topk(k)

            # for each live hypothesis explore k alternatives
            for i, k_bests in enumerate(zip(probs, indeces)):
                best_probs, best_indeces = k_bests
                hypothesis = live_hypotheses[i]

                for log_prob, index in zip(best_probs, best_indeces):
                    # create a new hypothesis for each k alternative

                    new_hyp = deepcopy(hypothesis)
                    token_id = index

                    # update hypothesis with new word and score
                    new_hyp.update(
                        token_id, torch.zeros(1), log_prob.item()
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
