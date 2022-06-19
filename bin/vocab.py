"""
reads a file and creates a vocuabulary file
(TSV: word count). Words are listed in descending
order. Minimum frequency can be enforced.
"""

import sys

from utils.lang import Vocab


def main(args):
    if args.progress is True:
        sys.stderr.write("Starting: Building vocabulary\n")

    voc = Vocab(args.min_freq)
    for i, line in enumerate(sys.stdin):
        voc.add_sentence(line.strip())

        if args.progress is True:
            if (i+1) % 100000 == 0:
                sys.stderr.write(f"Processed lines: {i + 1:,}\n")

        if args.n_sample != 0:
            if i > args.n_sample:
                break

    for word, count in voc.get_vocab():
        sys.stdout.write(f"{word}\t{count}\n")

    if args.progress is True:
        sys.stderr.write("Complete: Building vocabulary\n")
