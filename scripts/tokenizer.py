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

sys.path.append("../preprocessing_tools")
sys.path.append("preprocessing_tools")

from sentence_piece import SentencePieceTokenizer


def main(args):
    if args.input is not None:
        input_stream = Path(args.input).open(encoding="utf-8")
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        output_file = sys.stdout

    model = Path(args.model)

    if model.is_file():
        tokenizer = SentencePieceTokenizer.from_pretrained(model)
    else:
        tokenizer = SentencePieceTokenizer(args.size)

        if args.input is None:
            tmp = tempfile.TemporaryFile("w+", encoding="utf-8")
            for line in input_stream:
                tmp.write(line)

            tmp.seek(0)
            input_stream = tmp

        tokenizer.train(input_stream, args.bpe, args.train_lines)
        tokenizer.save_model(args.model)
        input_stream.seek(0)

    for i, line in enumerate(input_stream):
        print(tokenizer(line), file=output_file)

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
        "--train-lines",
        action="store",
        metavar="N",
        help="number of lines used for training (default: %(default)s)",
        type=int,
        default=1000000,
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
