"""
translate a text file with a pretrained model
"""

from pathlib import Path
import os

import torch

from model.seq2seq import seq2seq


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

    # transfer model and set eval mode
    model.to(device)
    model.eval()

    # print model
    print(model)

    # start translating
    outputfile = Path(f"data/translated.{model.tgt_lang.name}")
    with open(inputfile, "r", encoding="utf-8") as infile, \
            open(outputfile, "w", encoding="utf-8") as outfile:
        for progress, line in enumerate(infile):
            line = line.strip()
            hypotheses = model.beam_search(line, beam_size, device)

            # if verbose print all hypotheses
            if args.verbose:
                for hyp in hypotheses:
                    indeces = hyp.get_indeces()
                    tokens = [
                        model.tgt_lang.index2word[i.item()] for i in indeces
                    ]
                    print(tokens)

            # get indeces of best hypothesis
            indeces = hypotheses[0].get_indeces()
            tokens = [model.tgt_lang.index2word[i.item()] for i in indeces]

            # remove SOS and EOS
            tokens = tokens[1:-1]
            translated = " ".join(tokens)

            # write decoded sentence to output file
            outfile.write(f"{translated}\n")

            if args.verbose:
                print()
            else:
                print(f"Translating: line {progress + 1}", end="\r")

    print("Translating: complete")
