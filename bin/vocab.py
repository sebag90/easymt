"""
reads a file and creates a vocuabulary file
(TSV: word count). Words are listed in descending
order. Minimum frequency can be enforced.
"""

import sys
import pickle
import io
from utils.lang import Vocab

from preprocessing_tools.pipeline import Pipeline


def main(args):
    print("Starting: Building vocabulary", file=sys.stderr)
    first_line = sys.stdin.buffer.readline()

    try:
        # input is a text file, count the words
        first_line = first_line.decode()
        voc = Vocab(args.min_freq)
        voc.add_sentence(first_line.strip())

        for i, line in enumerate(sys.stdin):
            voc.add_sentence(line.strip())

            if args.progress is True:
                if (i+1) % 100000 == 0:
                    print(f"Processed lines: {i + 1:,}", file=sys.stderr)

            if args.n_sample != 0:
                if i + 1 > args.n_sample:
                    break

        for word, count in voc.get_vocab():
            print(f"{word}\t{count}", file=sys.stdout)

    except UnicodeDecodeError:
        # input file is a preprocessing model, extract vocab
        model = first_line + sys.stdin.buffer.read()
        data = pickle.loads(model)

        if data["pipe"] == ["sp"]:
            for word, score in data["tools"].sp.get_vocab():
                print(f"{word}\t{score}", file=sys.stdout)
        else:
            print(
                "Preprocessing model contains no vocabulary, "
                "extract it from the text file instead"
            )

    print("Complete: Building vocabulary", file=sys.stderr)
