"""
The decoding function revert tokenization
and remove subword splitting if needed
"""

from io import TextIOWrapper
from pathlib import Path
import sys
import pickle

from preprocessing_tools.pipeline import Pipeline


def main(args):
    print("Starting: Normalization", file=sys.stderr)

    modelpath = Path(args.model)

    with modelpath.open("rb") as infile:
        model = pickle.load(infile)

    pipe = Pipeline.from_trained_model(model)

    # start decoding
    input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    for i, line in enumerate(input_stream):
        line = pipe.decode(line)

        if args.upper is True:
            line = line.capitalize()

        print(line, file=sys.stdout)

        if (i+1) % 10000 == 0:
            print(f"Processed lines: {i + 1:,}", file=sys.stderr)

    print("Complete: Normalization", file=sys.stderr)
