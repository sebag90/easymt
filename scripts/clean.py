import argparse
from io import TextIOWrapper
from pathlib import Path
import sys

# append preprocessing tools both if run from
# easymt root directory or script directory
sys.path.append("../preprocessing_tools")
sys.path.append("preprocessing_tools")

from dexmler import Dexmler
from cleaner import Cleaner


def main(args):
    print("Starting: Cleaning", file=sys.stderr)

    dexmler = Dexmler()
    cleaner = Cleaner(
        min_len=args.min_len,
        max_len=args.max_len,
        ratio=args.ratio
    )

    if args.input is not None:
        input_stream = Path(args.input).open(encoding="utf-8")
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        output_file = sys.stdout

    for i, line in enumerate(input_stream):
        lines = line.split("\t")

        # clean files
        dexmled = dexmler(*lines)
        cleaned = cleaner(*dexmled)

        if all((i != "" for i in cleaned)):
            print("\t".join(cleaned), file=output_file)

        # print progress
        if i % 1000000 == 0:
            print(i, end="", file=sys.stderr, flush=True)

        elif i % 100000 == 0:
            print(".", end="", file=sys.stderr, flush=True)

    print("\nComplete: Cleaning", file=sys.stderr)
    input_stream.close()
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-len",
        metavar="N",
        action="store",
        help="maximum sentence length (default: %(default)s)",
        default=256,
        type=int
    )
    parser.add_argument(
        "--min-len",
        action="store",
        help="minimum sentence length (default: %(default)s)",
        default=1,
        type=int
    )
    parser.add_argument(
        "--ratio",
        action="store",
        help="maximum ratio between length of sources (default: %(default)s)",
        default=9,
        type=int
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        action="store",
        required="--output" in sys.argv
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        action="store",
        required="--input" in sys.argv
    )

    args = parser.parse_args()
    main(args)
