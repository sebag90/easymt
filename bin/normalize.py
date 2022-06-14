"""
The normalizing function revert tokenization
and remove subword splitting if needed
"""

from pathlib import Path
import re

import sentencepiece as spm

from utils.utils import split_filename
from preprocessing_tools.detokenizer import Detokenizer
from preprocessing_tools.truecaser import Truecaser


def main(args):
    print("Starting: Normalization")

    path, name, suffix = split_filename(args.file)
    ofile = Path(f"{path}/{name}.normalized.{suffix}")

    if args.SP is not None:
        sp_model = f"{path}/model.sentencepiece.{args.SP}.{suffix}"
        sp = spm.SentencePieceProcessor(model_file=sp_model)
    else:
        truecaser = Truecaser(suffix, path)
        detok = Detokenizer(suffix)
        subword_regex = re.compile(r"@@( |$)")

    with open(Path(args.file), "r", encoding="utf-8") as infile, \
            open(ofile, "w", encoding="utf-8") as ofile:
        for i, line in enumerate(infile):
            if args.SP is not None:
                # file was encoded with sentencepiece
                to_write = sp.decode(line.strip().split())
                to_write = to_write.replace("⁇", "<unk>")

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

            if (i+1) % 10000 == 0:
                print(f"Processed lines: {i + 1:,}", flush=True)

    print("Complete: Normalization")
