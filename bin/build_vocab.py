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
                if (i+1) % 100000 == 0:
                    print(f"Processed lines: {i + 1:,}", flush=True)

            if n_sample != 0:
                if i > n_sample:
                    break

    voc.save_voc(path)


def main(args):
    print("Starting: Building vocabulary")

    n_sample = int(args.n_sample)
    verbose = [False for _ in args.file]
    verbose[0] = True

    mp_args = list()
    for filename, verb in zip(args.file, verbose):
        mp_args.append(tuple([Path(filename), n_sample, args.min_freq, verb]))

    mp.set_start_method("spawn")
    with mp.Pool() as pool:
        pool.starmap(process_single_file, mp_args)

    print("Complete: Building vocabulary")
