"""
The preprocessing step prepares clean data to be used
for machine translation. The pipeline will:
    - normalize punctuation
    - tokenize
    - truecase
    - apply subword splitting (optional)
"""

import os
from pathlib import Path
import time
import datetime

from preprocessing_tools.tokenizer import Tokenizer
from preprocessing_tools.truecaser import Truecaser
from preprocessing_tools.punct_normalizer import PunctNormalizer
from preprocessing_tools.subword_splitter import SubwordSplitter

from utils.utils import name_suffix_from_file


class LowerCaser:
    trained = False

    def __repr__(self):
        return "Lowercaser"

    def __call__(self, line):
        return line.lower().strip()

    def train(self, path):
        pass


class Pipeline:
    def __init__(self, filename, language, bpe):
        self.filename = filename

        self.pipe = [
            PunctNormalizer(language),
            Tokenizer(language)
        ]

        self.trainable = [
            Truecaser(language),
            LowerCaser()  # must be applied after training Truecaser
        ]

        # add bpe splitter to trainable processors
        if bpe is not None:
            self.trainable.append(
                SubwordSplitter(language, bpe)
            )

    def apply_trainable(self, processor):
        # train processor on last produced text file
        processor.train(Path(f"data/temp.txt"))

        # do not apply truecaser
        if not isinstance(processor, Truecaser):
            # rename temp --> step_input
            os.rename(
                Path(f"data/temp.txt"), Path(f"data/step_input.txt")
            )

            # output is going to be temp.txt again
            input_name = Path(f"data/step_input.txt")
            output_name = Path(f"data/temp.txt")

            with open(input_name, "r", encoding="utf-8") as infile,\
                    open(output_name, "w", encoding="utf-8") as ofile:
                for i, line in enumerate(infile):
                    line = processor(line)
                    ofile.write(f"{line}\n")
                    print(f"Preprocessing: line {i}", end="\r")

            # remove step input
            os.remove(Path(f"data/step_input.txt"))

    def run(self):
        to_train = list()

        # collect trained processors
        for processor in self.trainable:
            if processor.trained:
                self.pipe.append(processor)
            else:
                to_train.append(processor)

        pipe_string = " > ".join([str(i) for i in self.pipe])
        complete = " > ".join([str(i) for i in self.pipe + to_train])
        print(f"Pipe: {complete}\n")
        t_0 = time.time()

        print(f"Applying: {pipe_string}")
        # applied untrainable and trained processors
        with open(Path(self.filename), "r", encoding="utf-8") as infile,\
                open(Path(f"data/temp.txt"), "w", encoding="utf-8") as ofile:
            for i, line in enumerate(infile):

                for processor in self.pipe:
                    if not isinstance(processor, Truecaser):
                        line = processor(line)

                ofile.write(f"{line}\n")
                print(f"Preprocessing: line {i}", end="\r")

        print(" " * 100, end="\r")
        t_1 = time.time()
        ts = int(t_1 - t_0)
        print(f"Timestamp: {datetime.timedelta(seconds=ts)}\n")

        # train and apply untrained processors
        for processor in to_train:
            print(f"Applying: {processor}")
            self.apply_trainable(processor)

            t_1 = time.time()
            ts = int(t_1 - t_0)
            print(" " * 100, end="\r")
            print(f"Timestamp: {datetime.timedelta(seconds=ts)}\n")

        name, suffix = name_suffix_from_file(self.filename)
        # rename last output file
        os.rename(Path(f"data/temp.txt"), Path(f"{name}_processed.{suffix}"))


def preprocess(args):
    pipe = Pipeline(args.file, args.language, args.bpe)
    pipe.run()
    print("Preprocessing: complete")
