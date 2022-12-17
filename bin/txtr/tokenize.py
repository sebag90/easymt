"""
The preprocessing step prepares clean data to be used
for machine translation. The pipeline will:
    - normalize punctuation
    - tokenize
    - truecase
    - apply subword splitting (optional)
"""

from io import TextIOWrapper
from pathlib import Path
import pickle
import sys

from preprocessing_tools.pipeline import Pipeline


def main(args):
    print("Starting: Preprocessing", file=sys.stderr)
    modelpath = Path(args.model)
    input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    if modelpath.is_file() is True:
        with modelpath.open("rb") as infile:
            model = pickle.load(infile)

        pipe = Pipeline.from_trained_model(model)
        pipe.run(input_stream)

    else:
        pipe = Pipeline(
            args.language,
            args.bpe,
            args.sp,
            args.replace_nums,
            args.max_lines,
            args.processors
        )
        pipe.run(input_stream)
        model = pipe.get_model()

        with modelpath.open("wb") as ofile:
            pickle.dump(model, ofile)

    print("Complete: Preprocessing", file=sys.stderr)
