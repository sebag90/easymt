"""
The preprocessing step prepares clean data to be used
for machine translation. The pipeline will:
    - normalize punctuation
    - tokenize
    - truecase
    - apply subword splitting (optional)
"""

import datetime
import io
import os
from pathlib import Path

import sys
import time

import sentencepiece as spm
from preprocessing_tools.pipeline import Pipeline


def main(args):
    print("Starting: Preprocessing", file=sys.stderr)

    if args.SP is None:
        pipe = Pipeline(
            args.language,
            args.bpe,
            args.replace_nums,
            args.max_lines
        )
        pipe.run(sys.stdin)
    else:
        # if model is already trained
        modelpath = Path(args.model)
        if modelpath.is_file():
            sp = spm.SentencePieceProcessor(
                model_file=args.model
            )
            read_from = sys.stdin

        else:
            # model needs to be trained
            model = io.BytesIO()
            t_file = tempfile.TemporaryFile(mode="w+")

            for line in sys.stdin:
                t_file.write(line)

            t_file.seek(0)

            # train and load model
            spm.SentencePieceTrainer.train(
                sentence_iterator=t_file,
                model_writer=model,
                vocab_size=args.SP,
                bos_id=-1,
                eos_id=-1,
                input_sentence_size=args.max_lines
            )
            sp = spm.SentencePieceProcessor(
                model_proto=model.getvalue()
            )
        
            # save model
            with open(modelpath, "wb") as ofile:
                ofile.write(model.getvalue())

            t_file.seek(0)
            read_from = t_file

        # tokenize file
        for i, line in enumerate(read_from):
            encoded = sp.encode(line.strip(), out_type=str)
            sys.stdout.write(f"{' '.join(encoded)}\n")
            if (i+1) % 100000 == 0:
                print(f"Processed lines: {i + 1:,}", file=sys.stderr)

    print("Complete: Preprocessing", file=sys.stderr)
