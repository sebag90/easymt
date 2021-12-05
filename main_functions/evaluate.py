"""
compute BLEU score using a reference translation
"""

from distutils.spawn import find_executable
import os
from pathlib import Path
import subprocess

from utils.errors import FileError


def evaluate(args):
    # make sure reference and translation have same length
    ref = 0
    with open(Path(args.reference), "r", encoding="utf-8") as rfile:
        for line in rfile:
            ref += 1

    tra = 0
    with open(Path(args.translation), "r", encoding="utf-8") as tfile:
        for line in tfile:
            tra += 1

    if ref != tra:
        raise FileError(
            "Translation and reference must have the same length"
        )

    # calculate BLEU with perl script
    os.system(
        "perl preprocessing-tools/multi-bleu-detok.perl "
        f" {args.reference} < {args.translation}"
    )

    # if sacrebleu is installed, run it on files
    sacrebleu_exists = find_executable("sacrebleu")
    if sacrebleu_exists is not None:
        print("\nSacreBLEU:")
        command = (
            f"sacrebleu {args.reference} -i {args.translation}"
        )
        result = subprocess.check_output(command, shell=True)
        result = eval(result)
        for key, value in result.items():
            print(f"{key:17}{value}")
