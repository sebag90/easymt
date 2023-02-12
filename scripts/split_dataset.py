"""
Split a single file in train, eval and test files
based on number of lines for each subset
"""
import argparse
from io import TextIOWrapper
from pathlib import Path
import sys


def main(args):
    if args.input is not None:
        input_stream = Path(args.input).open(encoding="utf-8")
    else:
        input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    limits = [args.train, args.eval, args.test]
    endings = ["train", "eval", "test"]
    o_index = 0

    outputfiles = list()
    for ending in endings:
        outputfiles.append(
            Path(f"{args.output}.{ending}").open("w", encoding="utf-8")
        )
    current_limit = limits[o_index]
    current_file = outputfiles[o_index]
    results = {name: 0 for name in endings}

    for i, line in enumerate(input_stream, start=0):
        current_file.write(line)
        results[endings[o_index]] += 1

        if i == current_limit:
            o_index += 1

            if o_index == len(limits):
                break

            current_limit += limits[o_index]
            current_file = outputfiles[o_index]

        if i % 1000000 == 0:
            print(i, end="", file=sys.stderr, flush=True)

        elif i % 100000 == 0:
            print(".", end="", file=sys.stderr, flush=True)

    for out_file in outputfiles:
        out_file.close()
    input_stream.close()

    print("\nLines:", file=sys.stderr)
    for name, lines in results.items():
        print(name, lines, file=sys.stderr)
    print("Complete: Splitting data set", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "split a text file in .train, .eval and .test based on "
            "number of lines for each subset "
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        action="store",
        required=True,
        help=(
            "path to the output files (endings .train, "
            ".test and .eval will be added)"
        )
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        action="store",
    )
    parser.add_argument(
        "--train", metavar="N_train",
        action="store",
        type=int,
        help="number of lines for train data",
        required=True
    )
    parser.add_argument(
        "--eval", metavar="N_eval",
        action="store",
        type=int,
        help="number of lines for evaluation data",
        required=True
    )
    parser.add_argument(
        "--test", metavar="N_test",
        action="store",
        type=int,
        help="number of lines for test data",
        required=True
    )

    args = parser.parse_args()
    main(args)
