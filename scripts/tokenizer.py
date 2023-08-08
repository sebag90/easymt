"""
Train and apply a BPE tokenizer to preprocess data
"""

import argparse
from io import TextIOWrapper
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import uuid

from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    processors
)


class MultiTok:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def single(self, line):
        return " ".join(self.tokenizer.encode(line.strip()).tokens)

    def multi(self, input_stream, outfile):
        iter_file = (line for line in input_stream)

        with mp.Pool() as pool:
            for i, line in enumerate(pool.imap(self.single,
                                            iter_file,
                                            chunksize=300)):

                print(line, file=outfile)

                # print progress
                if i % 1000000 == 0:
                    print(i, end="", file=sys.stderr, flush=True)

                elif i % 100000 == 0:
                    print(".", end="", file=sys.stderr, flush=True)


def main(args):
    if args.input is not None:
        input_stream = Path(args.input).open(encoding="utf-8")
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        input_stream = TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
        output_file = sys.stdout

    if not Path(args.model).is_file():
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(
            trim_offsets=True
        )

        extra_tokens = args.special_tokens or list()

        trainer = trainers.BpeTrainer(
            vocab_size=args.size,
            min_frequency=args.min_freq,
            special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"] + extra_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        if args.input is None:
            tmp = tempfile.TemporaryFile("w+", encoding="utf-8")
            for line in input_stream:
                tmp.write(line)

            tmp.seek(0)
            input_stream = tmp

        if args.train_file is not None:
            tokenizer.train([args.train_file], trainer=trainer)
        else:
            tokenizer.train_from_iterator(
                (i.strip() for i in input_stream), trainer=trainer
            )

        tokenizer.save(args.model)
        input_stream.seek(0)

    tokenizer = Tokenizer.from_file(args.model)

    # start multiprocessing tokenization
    multitok = MultiTok(tokenizer)
    multitok.multi(input_stream, output_file)

    print("\nComplete: Tokenizing", file=sys.stderr)
    input_stream.close()
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "tokenize the input text based on a trained tokenizer. "
            "If the tokenizer.json does not exists, the script will "
            "first train a tokenizer and then use this to tokenize "
            "the input test"
        )
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        action="store",
        required="--output" in sys.argv
    )
    parser.add_argument(
        "--model",
        "-m",
        action="store",
        help=(
            "output path for the trained tokenizer "
            "or to an already existing tokenizer"
        ),
        required=True,
    )
    parser.add_argument(
        "--size",
        action="store",
        metavar="N",
        help="vocabulary size (default: %(default)s)",
        type=int,
        default=35000,
    )
    parser.add_argument(
        "--min-freq",
        action="store",
        metavar="N",
        help="minimum frequency of a token(default: %(default)s)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--train-file",
        action="store",
        metavar="PATH",
        help="path to a training file for the tokenizer",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        action="store",
        required="--input" in sys.argv
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
    )

    args = parser.parse_args()
    mp.set_start_method('spawn')
    main(args)
