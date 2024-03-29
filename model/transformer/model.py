from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lang import Hypothesis


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
        self.generator = nn.Linear(self.size, len(self.tgt_lang))

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
            f"{self.decoder}\n"
            f"Generator: (\n  {self.generator}\n)"
        )
        return obj_str

    def prepare_batch(self, batch):
        src, tgt = batch
        padding_value = self.src_lang.word2index["<pad>"]
        # create index tensors from token lists
        # decoder input == target sentence with <sos> and no <eos>
        decoder_input = [
            self.tgt_lang.toks2idx(sen, eos=False) for sen in tgt
        ]
        # source sentence only needs <eos>
        src = [self.src_lang.toks2idx(sen, sos=False) for sen in src]

        # target sentence has no <sos> (because it's the input of the decoder)
        # but has a <eos> (must be learnt from the decoder)
        tgt = [self.tgt_lang.toks2idx(sen, sos=False) for sen in tgt]

        # pad tensors
        decoder_input = nn.utils.rnn.pad_sequence(
            decoder_input,
            batch_first=True,
            padding_value=padding_value

        )
        src = nn.utils.rnn.pad_sequence(
            src,
            batch_first=True,
            padding_value=padding_value

        )
        target = nn.utils.rnn.pad_sequence(
            tgt,
            batch_first=True,
            padding_value=padding_value
        )

        # create masks
        e_mask = (src != padding_value).unsqueeze(1)
        d_mask = (decoder_input != padding_value).unsqueeze(1)
        subseq_mask = torch.ones(
            (1, decoder_input.size(1), decoder_input.size(1)),
            dtype=torch.bool
        )
        subseq_mask = torch.tril(subseq_mask)
        d_mask = torch.logical_and(d_mask, subseq_mask)

        return src, decoder_input, target, e_mask, subseq_mask

    def forward(self, batch, teacher_forcing_ratio, criterion):
        (input_var,
         decoder_input,
         target_var,
         e_mask,
         d_mask) = self.prepare_batch(batch)

        # move tensors to device
        input_var = input_var.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        target_var = target_var.to(DEVICE)
        e_mask = e_mask.to(DEVICE)
        d_mask = d_mask.to(DEVICE)

        # obtain embedded sequences
        input_var = self.embedding.src(input_var)
        decoder_input = self.embedding.tgt(decoder_input)

        # pass through encoder and decoder
        encoded = self.encoder(input_var, e_mask)
        output = self.decoder(decoder_input, encoded, e_mask, d_mask)
        decoded = self.generator(output)
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
    def beam_search(self, line, beam_size, alpha):
        """
        beam translation for a single line of text
        """
        # prepare input
        coded = self.src_lang.toks2idx(line.strip().split())
        padder = torch.nn.ZeroPad2d((0, self.max_len - coded.size(0)))
        src = padder(coded).unsqueeze(0)
        e_mask = (src != 0)
        src = src.to(DEVICE)
        e_mask = e_mask.to(DEVICE)

        # encode input sentence
        src = self.embedding.src(src)
        encoded = self.encoder(src, e_mask)

        complete_hypotheses = list()
        live_hypotheses = list()

        # create empty hypothesis with only <sos> token
        sos_index = self.src_lang.word2index["<sos>"]
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
            decoder_input = torch.vstack(step_input).to(DEVICE)
            encoder_output = torch.cat(encoder_output)

            # create mask for decoding
            d_mask = self.create_subsequent_mask(
                decoder_input.size(1)
            ).to(DEVICE)

            # pass through decoder
            decoder_input = self.embedding.tgt(decoder_input)
            output = self.decoder(
                decoder_input, encoder_output, e_mask, d_mask
            )
            decoded = self.generator(output)

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

            # increase time step
            t += 1

        # if no complete hypothesis, return the alive ones
        if len(complete_hypotheses) == 0:
            complete_hypotheses = live_hypotheses

        # pick most probabile hypothesis
        complete_hypotheses.sort(reverse=True)

        # return sorted hypotheses
        return complete_hypotheses
