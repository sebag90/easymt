"""
The normalizing function revert tokenization
and remove subword splitting if needed
"""

import os
from pathlib import Path
import re

from utils.utils import name_suffix_from_file
from preprocessing_tools.detokenizer import Detokenizer
from preprocessing_tools.truecaser import Truecaser


def normalize(args):
    full_name = args.file.split(os.sep)[-1]
    name, suffix = name_suffix_from_file(full_name)

    truecaser = Truecaser(suffix)
    detok = Detokenizer(suffix)
    ofile = Path(f"data/{name}.normalized.{suffix}")

    subword_regex = re.compile(r"@@( |$)")

    with open(Path(args.file), "r", encoding="utf-8") as infile, \
            open(ofile, "w", encoding="utf-8") as ofile:
        for i, line in enumerate(infile):
            # undo subword splitting
            if args.subword is True:
                line = re.sub(subword_regex, "", line)

            # truecase
            line = truecaser(line)

            # detokenize
            to_write = detok(line)

            # write output
            ofile.write(f"{to_write}\n")

            print(f"Normalizing: line {i}", end="\r")

    print(" "*100, end="\r")
    print("Normalizing: complete")
