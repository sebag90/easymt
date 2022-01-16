"""
Split a single file in train, eval and test files
based on number of lines for each subset
"""

import os
from pathlib import Path

from utils.utils import split_filename


def split_dataset(args):
    path1, name1, suffix1 = split_filename(args.l1)
    path2, name2, suffix2 = split_filename(args.l2)

    train_n = int(args.train)
    test_n = int(args.test)
    eval_n = int(args.eval)

    outputs = ["train", "eval", "test"]
    limits = [train_n, eval_n, test_n]
    o_index = 0
    limit = limits[o_index]

    with open(Path(args.l1), "r", encoding="utf-8") as srcvoc, \
            open(Path(args.l2), "r", encoding="utf-8") as tgtvoc:

        filename1 = Path(f"{path1}/{outputs[o_index]}.{suffix1}")
        filename2 = Path(f"{path2}/{outputs[o_index]}.{suffix2}")
        ofile1 = open(filename1, "w", encoding="utf-8")
        ofile2 = open(filename2, "w", encoding="utf-8")

        for i, (l1, l2) in enumerate(zip(srcvoc, tgtvoc)):
            ofile1.write(l1)
            ofile2.write(l2)

            if i == limit - 1:
                # close files
                ofile1.close()
                ofile2.close()

                # increase index
                o_index += 1

                if o_index == len(outputs):
                    break

                # increase limit
                limit += limits[o_index]

                # open new files
                filename1 = Path(f"{path1}/{outputs[o_index]}.{suffix1}")
                filename2 = Path(f"{path2}/{outputs[o_index]}.{suffix2}")
                ofile1 = open(filename1, "w", encoding="utf-8")
                ofile2 = open(filename2, "w", encoding="utf-8")

            print(f"Splitting dataset: line {i:,}", end="\r")

    print(" "*50, end="\r")
    print("Splitting dataset: complete")
