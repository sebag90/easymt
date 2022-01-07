"""
reads a file and creates a vocuabulary file
(TSV: word count). Words are listed in descending
order. Minimum frequency can be enforced.
"""

from pathlib import Path

from utils.lang import Vocab
from utils.utils import split_filename


def build_vocab(args):
    n_sample = int(args.n_sample)
    l1_file = Path(f"{args.file1}")
    l2_file = Path(f"{args.file2}")

    # get suffixes from filenames
    path1, name1, suff1 = split_filename(args.file1)
    path2, name2, suff2 = split_filename(args.file2)

    # instantiate vocab objects
    voc1 = Vocab(suff1, args.min_freq)
    voc2 = Vocab(suff2, args.min_freq)

    with open(l1_file, "r", encoding="utf-8") as srcvoc, \
            open(l2_file, "r", encoding="utf-8") as tgtvoc:
        for i, (l1, l2) in enumerate(zip(srcvoc, tgtvoc)):
            # add line to vocabs
            voc1.add_sentence(l1.strip())
            voc2.add_sentence(l2.strip())

            print(f"Building vocabulary: line {i + 1:,}", end="\r")

            if n_sample != 0:
                if i > n_sample:
                    break

    voc1.save_voc(path1)
    voc2.save_voc(path2)

    print(" "*50, end="\r")
    print("Building vocabulary: complete")
