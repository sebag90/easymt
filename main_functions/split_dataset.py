import os
from pathlib import Path
import re

from utils.errors import FileError


def obtain_name(filename):
    name = re.match(r"(.*)\.", filename).group(1)
    suffix = re.search(r"\.(.*)$", filename).group(1)
    return name, suffix



def split_dataset(args):
    name_l1_full = args.l1.split(os.sep)[-1]
    name_l2_full = args.l2.split(os.sep)[-1]

    name1, suffix1 = obtain_name(name_l1_full)
    name2, suffix2 = obtain_name(name_l2_full)
    
    train_n = int(args.train)
    test_n = int(args.test)
    eval_n = int(args.eval)

    # make sure files have same length
    l1 = 0
    with open(Path(args.l1), "r", encoding="utf-8") as infile:
        for _ in infile:
            l1 += 1

    l2 = 0
    with open(Path(args.l2), "r", encoding="utf-8") as infile:
        for _ in infile:
            l2 += 1

    if l1 != l2:
        raise FileError(
            "Corpus files must have the same length."
        )

    # make sure train, test and eval can be extracted from files    
    if train_n + eval_n + test_n > l1:
        raise ValueError(
            "Files are too short for selected splitting."
        )

    outputs = ["train", "eval", "test"]
    limits = [train_n, eval_n, test_n]
    o_index = 0
    limit = limits[o_index]


    with open(Path(args.l1), "r", encoding="utf-8") as srcvoc, \
            open(Path(args.l2), "r", encoding="utf-8") as tgtvoc:

        ofile1 = open(Path(f"data/{outputs[o_index]}.{suffix1}"), "w", encoding="utf-8")
        ofile2 = open(Path(f"data/{outputs[o_index]}.{suffix2}"), "w", encoding="utf-8")

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
                ofile1 = open(Path(f"data/{outputs[o_index]}.{suffix1}"), "w", encoding="utf-8")
                ofile2 = open(Path(f"data/{outputs[o_index]}.{suffix2}"), "w", encoding="utf-8")

            print(f"Splitting dataset: line {i}", end="\r")

    print(" "*100, end="\r")
    print("Splitting dataset: complete")

