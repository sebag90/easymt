"""
The decoding function revert tokenization
and remove subword splitting if needed
"""

from pathlib import Path
import sys
import pickle

from preprocessing_tools.pipeline import Pipeline


def main(args):
    print("Starting: Normalization", file=sys.stderr)

    modelpath = Path(args.model)

    with open(modelpath, "rb") as infile:
        model = pickle.load(infile)

    pipe = Pipeline.from_trained_model(model)

    # start decoding
    for i, line in enumerate(sys.stdin):
        line = pipe.decode(line)

        if args.upper is True:
            line = line.capitalize()

        print(line, file=sys.stdout)

        if (i+1) % 10000 == 0:
            print(f"Processed lines: {i + 1:,}", file=sys.stderr)

    print("Complete: Normalization", file=sys.stderr)
