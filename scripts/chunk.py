import argparse
from io import TextIOWrapper
import sys
from pathlib import Path


def slice_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main(args):
    if args.input is not None:
        input_stream = Path(args.input).open(encoding="utf-8")
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        output_file = sys.stdout

    for i, line in enumerate(input_stream, start=1):
        chunks = slice_list(line.strip().split(), args.max_len)
        for sen in chunks:
            to_write = " ".join(sen)
            print(f"{to_write}", file=output_file)

        # print progress
        if i % 1000000 == 0:
            print(i, end="", file=sys.stderr, flush=True)

        elif i % 100000 == 0:
            print(".", end="", file=sys.stderr, flush=True)

    print("\nComplete: Chunking", file=sys.stderr)
    input_stream.close()
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "chunk sentences in a text file so that every line has a "
            "user defined maximum length "
        )
    )
    parser.add_argument(
        "--max-len",
        metavar="N",
        action="store",
        type=int,
        help="maximum length of each sentence (default: %(default)s)",
        required=True,
        default=1024
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
