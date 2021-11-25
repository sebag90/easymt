from copy import deepcopy
from functools import total_ordering
from pathlib import Path
import os

import torch
import torch.nn.functional as F

from model.seq2seq import seq2seq


@total_ordering
class Hypothesis:
    def __init__(self):
        self.score = 0
        self.tokens = []
        self.decoder_state = None

    def __str__(self):
        return str(self.score)

    def __repr__(self):
        return self.__str__()

    def add_word(self, word):
        self.tokens.append(word)

    def __lt__(self, other):
        return self.weigthed_score < other.weigthed_score

    @property
    def weigthed_score(self):
        return self.score / len(self.tokens)

    @property
    def last_word(self):
        return self.tokens[-1]

    @property
    def sentence(self):
        return self.__str__()

    def get_indeces(self):
        return torch.tensor(self.tokens)


def translate(args):
    inputfile = Path(args.file)
    model = seq2seq.load(Path(args.model))
    beam_size = int(args.beam)

    # pick device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cpu = os.cpu_count()
    torch.set_num_threads(cpu)

    # print model
    print(model)

    with open(inputfile, "r", encoding="utf-8") as infile, \
            open("raw.txt", "w", encoding="utf-8") as outfile:

        # get first 50 lines
        for progress, line in enumerate(infile):
            # encode the batch
            coded = model.src_lang.indexes_from_sentence(line)
            lengths = torch.tensor([len(coded)])
            input_batch = coded.unsqueeze(1)

            # encode sentence
            (decoder_input, *decoder_state, encoder_outputs) = model.encode(
                input_batch, lengths, device
            )

            complete_hypotheses = list()
            live_hypotheses = list()

            # create 5 empty hypotheses with only SOS token
            for i in range(beam_size):
                hyp = Hypothesis()
                hyp.add_word(decoder_input)
                hyp.decoder_state = decoder_state
                live_hypotheses.append(hyp)

            # begin beam search
            t = 0
            while t < model.max_len and len(complete_hypotheses) < beam_size:
                step_hypotheses = list()

                # iterate through hypotheses and explore them
                for hypothesis in live_hypotheses:
                    # obtain last decoder input from hypothesis
                    decoder_input = hypothesis.last_word
                    decoder_state = hypothesis.decoder_state

                    # pass through the decoder
                    decoder_output, *decoder_state = model.decoder_step(
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
                        new_hyp.add_word(decoded.unsqueeze(0).unsqueeze(0))
                        new_hyp.decoder_state = decoder_state
                        new_hyp.score += log_prob

                        # complete hypothesis if decoded EOS
                        idx = decoded.item()
                        if model.tgt_lang.index2word[idx] == "EOS":
                            complete_hypotheses.append(new_hyp)
                        else:
                            step_hypotheses.append(new_hyp)

                # prune the k best live_hypotheses
                step_hypotheses.sort(reverse=True)
                live_hypotheses = step_hypotheses[:k]

                # increase t
                t += 1

            # pick most probabile hypothesis
            complete_hypotheses.sort(reverse=True)
            indeces = complete_hypotheses[0].get_indeces()
            tokens = [model.tgt_lang.index2word[i.item()] for i in indeces]

            # remove SOS and EOS
            tokens = tokens[1:-1]
            as_string = " ".join(tokens)

            # write decoded sentence to output file
            outfile.write(f"{as_string}\n")

            print(f"Translating: line {progress + 1}", end="\r")

    out_lang = model.tgt_lang.name

    if model.bpe != 0:
        # undo bpe splitting
        os.rename("raw.txt", "temp.txt")
        os.system(f"sed -r 's/(@@ )|(@@ ?$)//g' < temp.txt > raw.txt")
        os.remove("temp.txt")

    # detokenize
    os.system(
        f"perl preprocessing-tools/detokenizer.perl -u -l {out_lang} "
        f"< raw.txt > translated.{out_lang}"
    )

    os.remove("raw.txt")
