"""
translate a text file with a pretrained model
"""

from pathlib import Path
import os

import torch

from utils.utils import split_filename


def main(args):
    inputfile = Path(args.file)
    beam_size = int(args.beam)

    # pick device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cpu = os.cpu_count()
    torch.set_num_threads(cpu)

    checkpoint = torch.load(Path(args.model), map_location=device)
    model = checkpoint["model"]

    # transfer model and set eval mode
    model.to(device)
    model.eval()

    # print model
    print(model)

    path, name, suffix = split_filename(str(inputfile))

    # start translating
    outputfile = Path(f"{path}/{name}.translated.{model.tgt_lang.name}")
    with open(inputfile, "r", encoding="utf-8") as infile, \
            open(outputfile, "w", encoding="utf-8") as outfile:
        for progress, line in enumerate(infile):
            line = line.strip()
            hypotheses = model.beam_search(line, beam_size, device, args.alpha)

            # if verbose print all hypotheses
            if args.verbose:
                for hyp in hypotheses:
                    indeces = hyp.get_indeces()
                    tokens = model.tgt_lang.idx2toks(indeces.tolist())
                    print(tokens)

            # get indeces of best hypothesis
            indeces = hypotheses[0].get_indeces()
            tokens = model.tgt_lang.idx2toks(indeces.tolist())

            # remove SOS and EOS
            tokens = filter(lambda x: x not in {"<eos>", "<sos>"}, tokens)
            translated = " ".join(tokens)

            # write decoded sentence to output file
            outfile.write(f"{translated}\n")

            if args.verbose:
                print()
            else:
                print(f"Translating: line {progress + 1:,}", end="\r")

    print(" " * 50, end="\r")
    print("Translating: complete")
