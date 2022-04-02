"""
reads a file and creates a vocuabulary file
(TSV: word count). Words are listed in descending
order. Minimum frequency can be enforced.
"""

import multiprocessing as mp
from pathlib import Path

from utils.lang import Vocab
from utils.utils import split_filename


def process_single_file(filename, n_sample, min_freq, verbose=False):
    """
    single core processing function
    """
    path, name, suff = split_filename(str(filename))
    voc = Vocab(suff, min_freq)

    with open(filename, "r", encoding="utf-8") as inputfile:
        for i, line in enumerate(inputfile):
            # add line to vocabs
            voc.add_sentence(line.strip())

            if verbose is True:
                print(f"Building vocabulary: line {i + 1:,}", end="\r")

            if n_sample != 0:
                if i > n_sample:
                    break

    voc.save_voc(path)


def build_vocab(args):
    n_sample = int(args.n_sample)
    l1_file = Path(f"{args.file1}")
    l2_file = Path(f"{args.file2}")

    mp_args = [
        (l1_file, n_sample, args.min_freq, True),
        (l2_file, n_sample, args.min_freq, False)
    ]

    mp.set_start_method("spawn")
    with mp.Pool() as pool:
        pool.starmap(process_single_file, mp_args)
   

    print(" "*50, end="\r")
    print("Building vocabulary: complete")
