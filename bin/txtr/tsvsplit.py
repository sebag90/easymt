"""
split a TSV document containing 2 languages
into 2 documents containing each a single language
"""

from pathlib import Path
import sys


def main(args):
    print("Starting: Splitting document", file=sys.stderr)

    file1 = Path(f"{args.output}.l1")
    file2 = Path(f"{args.output}.l2")

    with file1.open("w", encoding="utf-8") as f1, \
            file2.open("w", encoding="utf-8") as f2:
        for i, line in enumerate(sys.stdin, start=1):
            l1, l2 = line.strip().split("\t")

            f1.write(f"{l1}\n")
            f2.write(f"{l2}\n")

            if i % 100000 == 0:
                print(f"Processed lines: {i:,}", file=sys.stderr)

    print("Complete: Splitting document", file=sys.stderr)
