from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lang import Hypothesis


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_lang, tgt_lang, max_len):
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

    def forward(self, batch, device, teacher_forcing_ratio, criterion):
        input_var, decoder_input, target_var, e_mask, d_mask = batch

        # move tensors to device
        input_var = input_var.to(device)
        decoder_input = decoder_input.to(device)
        target_var = target_var.to(device)
        e_mask = e_mask.to(device)
        d_mask = d_mask.to(device)

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
    def beam_search(self, line, beam_size, device):
        """
        beam translation for a single line of text
        """
        # encode line
        coded = self.src_lang.indexes_from_sentence(line.strip())

        padder = torch.nn.ZeroPad2d((0, self.max_len - coded.size(0)))
        src = padder(coded).unsqueeze(0)
        e_mask = (src != 0)

        encoded = self.encoder(src, e_mask)

        # prepare decoder input
        sos_index = self.src_lang.word2index["<sos>"]
        d_mask = self.create_subsequent_mask(self.max_len)

        complete_hypotheses = list()
        live_hypotheses = list()

        # create empty hypothesis with only <sos> token
        hyp = Hypothesis()
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
            decoder_input = torch.vstack(step_input)
            encoder_output = torch.cat(encoder_output)

            # create mask for decoding
            d_mask = self.create_subsequent_mask(decoder_input.size(1))

            # pass through decoder
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
