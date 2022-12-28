"""
reads a file and creates a vocuabulary file
(TSV: word count). Words are listed in descending
order. Minimum frequency can be enforced.
"""

import sys
from utils.lang import Vocab
from pathlib import Path
import json

from tokenizers import Tokenizer


def main(args):
    input_file = Path(args.input)

    if args.output is not None:
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        output_file = sys.stdout

    try:
        model = json.loads(Path(args.input).read_text())
        tokenizer = Tokenizer.from_str(args.input)

        for word, position in sorted(tokenizer.get_vocab().items(), key=lambda x: x[-1]):
            print(f"{word}\t{position}", file=output_file)

    except JSONDecodeError:
        # input file is a text file, calculate vocabulary
        voc = Vocab(args.min_freq)
        with input_file.open(encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                voc.add_sentence(line.strip())

                # print progress
                if i % 1000000 == 0:
                    print(i, end="", file=sys.stderr, flush=True)

                elif i % 100000 == 0:
                    print(".", end="", file=sys.stderr, flush=True)

                if args.n_sample != 0:
                    if i > args.n_sample:
                        break

        for word, count in voc.get_vocab():
            print(f"{word}\t{count}", file=output_file)

    output_file.close()
    print("Complete: Building vocabulary", file=sys.stderr)
