"""
Split a single file in train, eval and test files
based on number of lines for each subset
"""

from io import TextIOWrapper
from pathlib import Path
import sys


def main(args):
    print("Starting: Splitting data set", file=sys.stderr)

    limits = [args.train, args.eval, args.test]
    endings = ["train", "eval", "test"]
    o_index = 0

    outputfiles = list()
    for ending in endings:
        outputfiles.append(
            open(Path(f"{args.output}.{ending}"), "w", encoding="utf-8")
        )
    current_limit = limits[o_index]
    current_file = outputfiles[o_index]
    results = {name: 0 for name in endings}

    input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    for i, line in enumerate(input_stream, start=1):
        current_file.write(line)
        results[endings[o_index]] += 1

        if i == current_limit:
            o_index += 1

            if o_index == len(limits):
                break

            current_limit += limits[o_index]
            current_file = outputfiles[o_index]

    for out_file in outputfiles:
        out_file.close()

    print("Lines:", file=sys.stderr)
    for name, lines in results.items():
        print(name, lines, file=sys.stderr)
    print("Complete: Splitting data set", file=sys.stderr)
