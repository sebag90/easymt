"""
Split a single file in train, eval and test files
based on number of lines for each subset
"""

import os
from pathlib import Path

from utils.utils import name_suffix_from_file


def split_dataset(args):
    name_l1_full = args.l1.split(os.sep)[-1]
    name_l2_full = args.l2.split(os.sep)[-1]

    name1, suffix1 = name_suffix_from_file(name_l1_full)
    name2, suffix2 = name_suffix_from_file(name_l2_full)

    train_n = int(args.train)
    test_n = int(args.test)
    eval_n = int(args.eval)

    path_1 = args.l1.split(os.sep)[:-1]
    path_2 = args.l2.split(os.sep)[:-1]
    path_1 = os.sep.join(path_1)
    path_2 = os.sep.join(path_2)

    outputs = ["train", "eval", "test"]
    limits = [train_n, eval_n, test_n]
    o_index = 0
    limit = limits[o_index]

    with open(Path(args.l1), "r", encoding="utf-8") as srcvoc, \
            open(Path(args.l2), "r", encoding="utf-8") as tgtvoc:

        filename1 = Path(f"{path_1}/{outputs[o_index]}.{suffix1}")
        filename2 = Path(f"{path_2}/{outputs[o_index]}.{suffix2}")
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
                filename1 = Path(f"{path_1}/{outputs[o_index]}.{suffix1}")
                filename2 = Path(f"{path_2}/{outputs[o_index]}.{suffix2}")
                ofile1 = open(filename1, "w", encoding="utf-8")
                ofile2 = open(filename2, "w", encoding="utf-8")

            print(f"Splitting dataset: line {i:,}", end="\r")

    print(" "*50, end="\r")
    print("Splitting dataset: complete")
