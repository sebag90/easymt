import os
from pathlib import Path
import time
import datetime

from preprocessing_tools.tokenizer import Tokenizer
from preprocessing_tools.truecaser import Truecaser
from preprocessing_tools.punct_normalizer import PunctNormalizer
from preprocessing_tools.subword_splitter import SubwordSplitter


class Pipeline:
    def __init__(self, filename, language, bpe):
        self.tokenizer = Tokenizer(language)
        self.truecaser = Truecaser(language)
        self.normalizer = PunctNormalizer(language)
        self.subword_splitter = SubwordSplitter(language, bpe)
        self.filename = filename

    def apply_trainable(self, processor):
        # train processor on last produced text file
        processor.train(Path(f"data/temp.txt"))

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
        pipe = [
            self.normalizer,
            self.tokenizer,
        ]
        trainable = [
            self.truecaser,
            self.subword_splitter
        ]

        to_train = list()

        # collect trained processors
        for processor in trainable:
            if processor.trained:
                pipe.append(processor)
            else:
                to_train.append(processor)

        pipe_string = " --> ".join([str(i) for i in pipe])
        complete = " --> ".join([str(i) for i in pipe + to_train])
        print(f"Pipe: {complete}\n")
        t_0 = time.time()

        print(f"Applying: {pipe_string}")
        # applied untrainable and trained processors
        with open(Path(self.filename), "r", encoding="utf-8") as infile,\
                open(Path(f"data/temp.txt"), "w", encoding="utf-8") as ofile:
            for i, line in enumerate(infile):

                for processor in pipe:
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

        # rename last output file
        os.rename(Path(f"data/temp.txt"), Path(f"{self.filename}_processed"))


def preprocess(args):
    pipe = Pipeline(args.file, args.language, args.bpe)
    pipe.run()
    print("Preprocessing: complete")