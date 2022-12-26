"""
compute BLEU score using a reference translation
"""
import argparse
from pathlib import Path
import sys

from torchtext.data.metrics import bleu_score


def main(args):
    ref_file =  Path(args.reference)
    trans_file = Path(args.translation)

    with ref_file.open("r", encoding="utf-8") as rfile, \
            trans_file.open("r", encoding="utf-8") as tfile:
        
        if args.lc is True:
            reference = [[i.strip().lower().split()] for i in rfile]
            translation = [i.strip().lower().split() for i in tfile]
        else:
            reference = [[i.strip().split()] for i in rfile]
            translation = [i.strip().split() for i in tfile]
        
        score = bleu_score(translation, reference)
        print(f"Bleu score: {score*100}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        "-r",
        action="store",
        required=True,
        help="path to reference translation"
    )
    parser.add_argument(
        "--translation",
        "-t",
        action="store",
        required=True,
        help="path to translated document"
    )
    parser.add_argument(
        "-lc", action="store_true",
        help="compute BLEU on lowercased text"
    )
    args = parser.parse_args()
    main(args)
