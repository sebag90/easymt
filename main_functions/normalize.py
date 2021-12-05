"""
The normalizing function revert tokenization
and remove subword splitting if needed
"""

import os
from pathlib import Path
import re

from utils.utils import name_suffix_from_file
from preprocessing_tools.detokenizer import Detokenizer


def normalize(args):
    full_name = args.file.split(os.sep)[-1]
    name, suffix = name_suffix_from_file(full_name)

    detok = Detokenizer(suffix)
    ofile = Path(f"data/{name}.normalized.{suffix}")

    with open(Path(args.file), "r", encoding="utf-8") as infile, \
            open(ofile, "w", encoding="utf-8") as ofile:
        for i, line in enumerate(infile):
            # undo subword splitting
            if args.subword is True:
                line = re.sub(r"@@ ", "", line)

            # detokenize
            to_write = detok(line)

            # write output
            ofile.write(f"{to_write}\n")

            print(f"Normalizing: line {i}", end="\r")

    print("Normalizing: complete")
