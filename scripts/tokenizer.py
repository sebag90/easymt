"""
The preprocessing step prepares clean data to be used
for machine translation. The pipeline will:
    - normalize punctuation
    - tokenize
    - truecase
    - apply subword splitting (optional)
"""
import argparse
from io import TextIOWrapper
from pathlib import Path
import sys
import tempfile

from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    processors
)


def main(args):
    if args.input is not None:
        input_stream = Path(args.input).open(encoding="utf-8")
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        output_file = sys.stdout

    if Path(args.model).is_file():
        tokenizer = Tokenizer.from_file(args.model)

    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(
            trim_offsets=True
        )

        trainer = trainers.BpeTrainer(
            vocab_size=args.size,
            min_frequency=args.min_freq,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        if args.input is None:
            tmp = tempfile.TemporaryFile("w+", encoding="utf-8")
            for line in input_stream:
                tmp.write(line)

            tmp.seek(0)
            input_stream = tmp

        if args.train_file is not None:
            tokenizer.train(
                [args.train_file],
                trainer=trainer
            )
        else:
            tokenizer.train_from_iterator(
                (i.strip() for i in input_stream),
                trainer=trainer
            )

        tokenizer.save(args.model)
        input_stream.seek(0)

    for i, line in enumerate(input_stream):
        tokens = " ".join(tokenizer.encode(line.strip()).tokens)
        print(tokens, file=output_file)

        # print progress
        if i % 1000000 == 0:
            print(i, end="", file=sys.stderr, flush=True)

        elif i % 100000 == 0:
            print(".", end="", file=sys.stderr, flush=True)

    print("\nComplete: Tokenizing", file=sys.stderr)
    input_stream.close()
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        required=True
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
        "--bpe",
        action="store_true",
        help="use BPE model",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        action="store",
        required="--input" in sys.argv
    )

    args = parser.parse_args()
    main(args)
