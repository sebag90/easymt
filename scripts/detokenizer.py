"""
The decoding function revert tokenization
and remove subword splitting if needed
"""
import argparse
from io import TextIOWrapper
from pathlib import Path
import sys

from tokenizers import Tokenizer


def main(args):
    tokenizer = Tokenizer.from_file(args.model)

    if args.input is not None:
        input_stream = Path(args.input).open(encoding="utf-8")
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        output_file = sys.stdout

    for i, line in enumerate(input_stream):
        line = line.strip().split()
        line = tokenizer.decode([tokenizer.token_to_id(i) for i in line])

        if args.upper is True:
            line = line.capitalize()

        print(line, file=output_file)

        # print progress
        if i % 1000000 == 0:
            print(i, end="", file=sys.stderr, flush=True)

        elif i % 100000 == 0:
            print(".", end="", file=sys.stderr, flush=True)

    print("\nComplete: Detokenization", file=sys.stderr)
    input_stream.close()
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        action="store",
        help="preprocessing model to decode text",
        required=True
    )
    parser.add_argument(
        "--upper",
        "-u",
        action="store_true",
        help="uppercase the first char in the sentence"
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
