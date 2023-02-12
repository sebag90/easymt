"""
joins 2 files in a TSV document
"""
import argparse
from pathlib import Path
import sys


def main(args):
    if args.output is not None:
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        output_file = sys.stdout

    file1, file2 = args.files

    file1 = Path(file1)
    file2 = Path(file2)

    with file1.open(encoding="utf-8") as f1, \
            file2.open(encoding="utf-8") as f2:

        for i, lines in enumerate(zip(f1, f2)):
            print("\t".join(i.strip() for i in lines), file=output_file)

            # print progress
            if i % 1000000 == 0:
                print(i, end="", file=sys.stderr, flush=True)

            elif i % 100000 == 0:
                print(".", end="", file=sys.stderr, flush=True)

    print("\nComplete: Splitting document", file=sys.stderr)
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "join different files into a single TSV file"
        )
    )
    parser.add_argument(
        "files",
        nargs=2,
        action="store",
        help="path to file(s) to be joined"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        action="store",
    )

    args = parser.parse_args()
    main(args)
