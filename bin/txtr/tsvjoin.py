"""
joins 2 files in a TSV document
"""

from pathlib import Path
import sys


def main(args):
    print("Starting: Splitting document", file=sys.stderr)

    files = [
        Path(filename).open() for filename in args.files
    ]

    for i, lines in enumerate(zip(*files), start=1):
        print("\t".join(i.strip() for i in lines), file=sys.stdout)

        if i % 100000 == 0:
            print(f"Processed lines: {i:,}", file=sys.stderr)

    print("Complete: Splitting document", file=sys.stderr)
