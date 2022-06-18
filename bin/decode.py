"""
The normalizing function revert tokenization
and remove subword splitting if needed
"""

from pathlib import Path
import sys
import re
import pickle

import sentencepiece as spm

from utils.utils import split_filename
from preprocessing_tools.detokenizer import Detokenizer
from preprocessing_tools.truecaser import Truecaser


def main(args):
    print("Starting: Normalization", file=sys.stderr)

    modelpath = Path(args.model)

    with open(modelpath, "rb") as infile:
        model = pickle.load(infile)

    # load model
    if model["type"] == "sp":
        sp = spm.SentencePieceProcessor(
                model_proto=model["model"]
            )
    else:
        for processor in model["pipe"]:
            if isinstance(processor, Truecaser):
                truecaser = processor
                break
        detok = Detokenizer(model["language"])
        subword_regex = re.compile(r"@@( |$)")

    # start decoding
    for i, line in enumerate(sys.stdin):
        if model["type"] == "sp":
            # file was encoded with sentencepiece
            to_write = sp.decode(line.strip().split())
            to_write = to_write.replace("⁇", "<unk>")

        else:
            # undo subword splitting
            if model["bpe"] > 0:
                line = re.sub(subword_regex, "", line)

            # truecase
            line = truecaser(line)

            # detokenize
            to_write = detok(line)

        if args.upper is True:
            to_write = to_write.capitalize()

        # write output
        sys.stdout.write(f"{to_write}\n")

        if (i+1) % 10000 == 0:
            print(f"Processed lines: {i + 1:,}", file=sys.stderr)

    print("Complete: Normalization", file=sys.stderr)
