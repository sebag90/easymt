"""
split a TSV document containing 2 languages
into 2 documents containing each a single language
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

    file1 = Path(f"{args.output}.l1")
    file2 = Path(f"{args.output}.l2")

    with file1.open("w", encoding="utf-8") as f1, \
            file2.open("w", encoding="utf-8") as f2:
        for i, line in enumerate(input_stream):
            l1, l2 = line.strip().split("\t")

            f1.write(f"{l1}\n")
            f2.write(f"{l2}\n")

            # print progress
            if i % 1000000 == 0:
                print(i, end="", file=sys.stderr, flush=True)

            elif i % 100000 == 0:
                print(".", end="", file=sys.stderr, flush=True)

    print("Complete: Splitting document", file=sys.stderr)
    input_stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        metavar="PATH",
        action="store"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        action="store",
        required=True
    )

    args = parser.parse_args()
    main(args)