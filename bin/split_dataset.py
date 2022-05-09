"""
Split a single file in train, eval and test files
based on number of lines for each subset
"""

import multiprocessing as mp
from pathlib import Path

from utils.utils import split_filename
from utils.errors import InvalidArgument


def split_single(filename, train_n, eval_n, test_n, verbose):
    path, name, suffix = split_filename(str(filename))
    to_write = ["train", "eval", "test"]
    outputfiles = list()
    for subpart in to_write:
        outputfiles.append(
            open(Path(f"{path}/{subpart}.{suffix}"), "w", encoding="utf-8")
        )

    limits = [train_n, eval_n, test_n]
    o_index = 0
    current_limit = limits[o_index]
    ofile = outputfiles[o_index]

    results = {f"{filename}.{suffix}": 0 for filename in to_write}

    with open(filename, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            ofile.write(line)
            results[f"{to_write[o_index]}.{suffix}"] += 1

            if i == current_limit - 1:
                o_index += 1

                if o_index == len(limits):
                    break

                current_limit += limits[o_index]
                ofile = outputfiles[o_index]

            if verbose is True:
                print(f"Splitting dataset: line {i:,}", end="\r")

    for out_file in outputfiles:
        out_file.close()

    return results


def main(args):
    # get arguments
    if len(args.file) < 1:
        raise InvalidArgument(
            "You need at least one file"
        )

    limits = [int(args.train), int(args.eval), int(args.test)]
    verbose = [False for _ in args.file]
    verbose[0] = True

    mp_args = list()
    for filename, verb in zip(args.file, verbose):
        mp_args.append(tuple([Path(filename), *limits, verb]))

    mp.set_start_method("spawn")
    with mp.Pool() as pool:
        splits = pool.starmap(split_single, mp_args)

    print(" "*79, end="\r")
    print("Lines:")
    for split in splits:
        print("\t".join([f"{key}: {value}" for key, value in split.items()]))

    print("Splitting dataset: complete")
