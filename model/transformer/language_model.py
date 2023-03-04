from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lang import Hypothesis


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class LanguageModel(nn.Module):
    def __init__(
            self, model, embedding, language, max_len, lm_type, size):
        super().__init__()
        self.type = lm_type
        self.model = model
        self.embedding = embedding
        self.src_lang = self.tgt_lang = language
        self.max_len = max_len
        self.steps = 0
        self.size = size
        self.generator = nn.Linear(size, len(self.tgt_lang))

    def __repr__(self):
        # count trainable parameters
        parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        # create print string
        obj_str = (
            f"LanguageModel({self.src_lang.name} | "
            f"steps: {self.steps:,} | "
            f"parameters: {parameters:,})\n"
            f"{self.embedding}\n"
            f"{self.model}\n"
            f"Generator: (\n  {self.generator}\n)"
        )
        return obj_str

    def prepare_batch(self, batch):
        # LM dataset only contains src
        sentence, _ = batch
        padding_value = self.src_lang.word2index["<pad>"]

        # convert tokens to indeces and:
        # - remove last word from source
        # - remove first word from target
        src = [
            self.src_lang.toks2idx(
                sen, sos=False, eos=False
            )[:-1] for sen in sentence
        ]
        tgt = [
            self.tgt_lang.toks2idx(
                sen, sos=False, eos=False
            )[1:] for sen in sentence
        ]

        # pad src and tgt
        tgt = nn.utils.rnn.pad_sequence(
            tgt,
            batch_first=True,
            padding_value=padding_value
        )
        src = nn.utils.rnn.pad_sequence(
            src,
            batch_first=True,
            padding_value=padding_value
        )

        # create masks
        e_mask = (src != padding_value).unsqueeze(1)
        d_mask = (tgt != padding_value).unsqueeze(1)
        subseq_mask = torch.ones(
            (1, tgt.size(1), tgt.size(1)),
            dtype=torch.bool
        )
        subseq_mask = torch.tril(subseq_mask)
        d_mask = torch.logical_and(d_mask, subseq_mask)
        return src, tgt, e_mask, subseq_mask

    def forward(self, batch, teacher_forcing_ratio, criterion):
        (input_var,
         target_var,
         e_mask,
         d_mask) = self.prepare_batch(batch)

        # move tensors to device
        input_var = input_var.to(DEVICE)
        target_var = target_var.to(DEVICE)
        e_mask = e_mask.to(DEVICE)
        d_mask = d_mask.to(DEVICE)

        # obtain embedded sequences
        input_var = self.embedding.src(input_var)

        if self.type == "encoder":
            mask = e_mask

        elif self.type == "generator":
            mask = d_mask


        # pass through model and generate logits
        output = self.model(input_var, mask.squeeze(0))
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
    def top_k(self, line, steps, k=10, temperature=1):
        step_input = self.src_lang.toks2idx(line.strip().split(), eos=False)
        hyp = Hypothesis(alpha=1)
        for word in step_input:
            hyp.update(
                word, torch.zeros(1), 0
            )
        step_input = step_input.unsqueeze(0).to(DEVICE)

        for _ in range(steps):
            # crop current context if it's too big
            if step_input.size(1) > self.max_len:
                step_input = step_input[-self.max_len:]

            # create sausal mask
            d_mask = self.create_subsequent_mask(
                step_input.size(1)
            ).to(DEVICE)

            # pass through decoder
            encoded = self.embedding.tgt(step_input)
            output = self.model(
                encoded, d_mask
            )
            decoded = self.generator(output)

            # divide logits by temperature
            decoded = decoded[:, -1, :] / temperature

            # select most k probabile words
            probs, idxs = torch.topk(decoded, k)

            # apply softmax and sample a word
            decoder_output = F.softmax(probs, dim=-1)
            idx_sample = torch.multinomial(
                decoder_output, num_samples=1
            ).item()
            idx_next = idxs[:, idx_sample]

            # update step input
            hyp.update(idx_next, torch.zeros(1), torch.zeros(1))
            step_input = torch.cat((step_input, idx_next.view(1, -1)), dim=1)

        return [hyp]

    @torch.no_grad()
    def beam_search(self, line, beam_size, alpha):
        """
        beam translation for a single line of text
        """
        # prepare input
        coded = self.src_lang.toks2idx(line.strip().split(), eos=False)

        complete_hypotheses = list()
        live_hypotheses = list()

        hyp = Hypothesis(alpha=alpha)
        for word in coded:
            hyp.update(
                word, torch.zeros(1), 0
            )
        live_hypotheses.append(hyp)

        # begin beam search
        t = 0
        while t < self.max_len and len(complete_hypotheses) < beam_size:
            step_hypotheses = list()

            # collect tensors from live hypotheses
            # to create input batch (batch size = beam size)
            step_input = list()

            # obtain input words
            for hypothesis in live_hypotheses:
                step_input.append(torch.tensor(hypothesis.tokens))

            # create batch with live hypotheses
            model_input = torch.vstack(step_input).to(DEVICE)

            # create mask for decoding
            d_mask = self.create_subsequent_mask(
                model_input.size(1)
            ).to(DEVICE)

            # pass through decoder
            model_input = self.embedding.tgt(model_input)
            output = self.model(
                model_input, d_mask
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
