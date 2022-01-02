"""
The normalizing function revert tokenization
and remove subword splitting if needed
"""

import os
from pathlib import Path
import re

import sentencepiece as spm

from utils.utils import split_filename
from preprocessing_tools.detokenizer import Detokenizer
from preprocessing_tools.truecaser import Truecaser


def normalize(args):
    full_name = args.file.split(os.sep)[-1]
    path, name, suffix = split_filename(full_name)

    ofile = Path(f"{path}/{name}.normalized.{suffix}")

    if args.sp_model is not None:
        sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    else:
        truecaser = Truecaser(suffix, path)
        detok = Detokenizer(suffix)
        subword_regex = re.compile(r"@@( |$)")

    with open(Path(args.file), "r", encoding="utf-8") as infile, \
            open(ofile, "w", encoding="utf-8") as ofile:
        for i, line in enumerate(infile):
            if args.sp_model is not None:
                # file was encoded with sentencepiece
                to_write = sp.decode(line.strip().split())

            else:
                # undo subword splitting
                if args.subword is True:
                    line = re.sub(subword_regex, "", line)

                # truecase
                line = truecaser(line)

                # detokenize
                to_write = detok(line)

            if args.upper is True:
                to_write = to_write.capitalize()

            # write output
            ofile.write(f"{to_write}\n")

            print(f"Normalizing: line {i:,}", end="\r")

    print(" "*50, end="\r")
    print("Normalizing: complete")
