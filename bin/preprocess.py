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
import pickle
import sys
import time

import sentencepiece as spm
from preprocessing_tools.pipeline import Pipeline


def main(args):
    print("Starting: Preprocessing", file=sys.stderr)
    modelpath = Path(args.model)

    if args.SP is None:
        pipe = Pipeline(
            args.language,
            args.bpe,
            args.replace_nums,
            args.max_lines
        )
        if modelpath.is_file() is True:
            with open(modelpath, "rb") as infile:
                model = pickle.load(infile)
            pipe.load_model(model)
            pipe.run(sys.stdin)

        else:
            pipe.run(sys.stdin)
            model = pipe.get_model()

            with open(modelpath, "wb") as ofile:
                pickle.dump(model, ofile)


    else:
        # if model is already trained
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
                json.dump(
                    {
                        "model": model.getvalue(),
                        "type": "sp"
                    },
                    ofile
                )

            t_file.seek(0)
            read_from = t_file

        # tokenize file
        for i, line in enumerate(read_from):
            encoded = sp.encode(line.strip(), out_type=str)
            sys.stdout.write(f"{' '.join(encoded)}\n")
            if (i+1) % 100000 == 0:
                print(f"Processed lines: {i + 1:,}", file=sys.stderr)

    print("Complete: Preprocessing", file=sys.stderr)
