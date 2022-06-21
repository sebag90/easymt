import sys

from preprocessing_tools.dexmler import Dexmler
from preprocessing_tools.cleaner import Cleaner


def main(args):
    print("Starting: Cleaning", file=sys.stderr)

    dexmler = Dexmler()
    cleaner = Cleaner(
        min_len=args.min_len,
        max_len=args.max_len,
        ratio=args.ratio
    )

    for i, line in enumerate(sys.stdin):
        lines = line.split("\t")

        # clean files
        dexmled = dexmler(*lines)
        cleaned = cleaner(*dexmled)

        if all((i != "" for i in cleaned)):
            print("\t".join(cleaned), file=sys.stdout)

        if (i+1) % 100000 == 0:
            print(f"Processed lines: {i + 1:,}", file=sys.stderr)

    print("Complete: Cleaning", file=sys.stderr)
