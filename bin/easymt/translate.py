"""
translate a text file with a pretrained model
"""

from io import TextIOWrapper
from pathlib import Path
import sys

import torch


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main(args):
    beam_size = int(args.beam)

    checkpoint = torch.load(Path(args.model), map_location=DEVICE)
    model = checkpoint["model"]

    # transfer model and set eval mode
    model.to(DEVICE)
    model.eval()

    # print model
    print(model, file=sys.stderr)

    # start translating
    input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    for progress, line in enumerate(input_stream):
        line = line.strip()
        
        if args.method == "beam":
            hypotheses = model.beam_search(line, beam_size, args.alpha)
        else:
            hypotheses = model.top_k(line, steps=args.steps, k=args.k, temperature=args.temperature)
        # if verbose print all hypotheses
        if args.verbose:
            for hyp in hypotheses:
                indeces = hyp.get_indeces()
                tokens = model.tgt_lang.idx2toks(indeces.tolist())
                print(tokens, file=sys.stderr)

        # get indeces of best hypothesis
        indeces = hypotheses[0].get_indeces()
        tokens = model.tgt_lang.idx2toks(indeces.tolist())

        # remove SOS and EOS
        tokens = filter(lambda x: x not in {"<eos>", "<sos>"}, tokens)
        translated = " ".join(tokens)

        # output translated sentence
        print(translated, file=sys.stdout)
        print(f"Translating: line {progress + 1:,}", file=sys.stderr)

    print("Translating: complete", file=sys.stderr)
