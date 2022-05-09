"""
The preprocessing step prepares clean data to be used
for machine translation. The pipeline will:
    - normalize punctuation
    - tokenize
    - truecase
    - apply subword splitting (optional)
"""

import datetime
import os
from pathlib import Path
import re
import time

import sentencepiece as spm

from preprocessing_tools.tokenizer import Tokenizer
from preprocessing_tools.truecaser import Truecaser
from preprocessing_tools.punct_normalizer import PunctNormalizer
from preprocessing_tools.subword_splitter import SubwordSplitter

from utils.utils import split_filename


class LowerCaser:
    def __init__(self, trained=False):
        self.trained = trained

    def __repr__(self):
        return "Lowercaser"

    def __call__(self, line):
        return line.lower().strip()

    def train(self, path):
        pass


class NumReplacer:
    num_token = "<num>"
    number = re.compile(r"(?<=\s)\d[\d,'.]*\b")

    def __call__(self, line):
        clean = re.sub(self.number, self.num_token, line)
        return clean.strip()

    def __repr__(self):
        return "NumReplacer"


class Pipeline:
    def __init__(self, filename, language, bpe, remove_nums, max_lines):
        self.filename = filename
        self.max_lines = max_lines
        path, name, suffix = split_filename(filename)
        self.path = path
        self.temp_file = Path(f"{path}/temp.{language}")
        self.language = language

        self.pipe = [
            PunctNormalizer(language),
            Tokenizer(language)
        ]

        if remove_nums is True:
            self.pipe.append(
                NumReplacer()
            )

        tr = Truecaser(language, path)
        lc = LowerCaser(tr.trained)

        self.trainable = [
            tr,
            lc  # must be applied after training Truecaser
        ]

        # add bpe splitter to trainable processors
        if bpe is not None:
            self.trainable.append(
                SubwordSplitter(language, bpe, path)
            )

    def apply_trainable(self, processor):
        # train processor on last produced text file
        if self.max_lines == 0:
            processor.train(self.temp_file)
        else:
            # only use max_lens lines to train
            trainfile = Path(f"{self.path}/train_processor.{self.language}")
            with open(self.temp_file, "r", encoding="utf-8") as infile,\
                    open(trainfile, "w", encoding="utf-8") as ofile:
                for i, line in enumerate(infile):
                    if i == self.max_lines:
                        break
                    ofile.write(line)

            # train on reduced file
            processor.train(trainfile)
            os.remove(trainfile)

        # do not apply truecaser
        if not isinstance(processor, Truecaser):
            # rename temp --> step_input
            os.rename(
                self.temp_file, Path(f"{self.path}/step_input.{self.language}")
            )

            # output is going to be temp.txt again
            input_name = Path(f"{self.path}/step_input.{self.language}")
            output_name = self.temp_file

            with open(input_name, "r", encoding="utf-8") as infile,\
                    open(output_name, "w", encoding="utf-8") as ofile:
                for i, line in enumerate(infile):
                    line = processor(line)
                    ofile.write(f"{line}\n")
                    print(f"Preprocessing: line {i:,}", end="\r")

            # remove step input
            os.remove(Path(f"{self.path}/step_input.{self.language}"))

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
                open(self.temp_file, "w", encoding="utf-8") as ofile:
            for i, line in enumerate(infile):

                for processor in self.pipe:
                    if not isinstance(processor, Truecaser):
                        line = processor(line)

                ofile.write(f"{line}\n")
                print(f"Preprocessing: line {i:,}", end="\r")

        print(" " * 50, end="\r")
        t_1 = time.time()
        ts = int(t_1 - t_0)
        print(f"Timestamp: {datetime.timedelta(seconds=ts)}\n")

        # train and apply untrained processors
        for processor in to_train:
            print(f"Applying: {processor}")
            self.apply_trainable(processor)

            t_1 = time.time()
            ts = int(t_1 - t_0)
            print(" " * 50, end="\r")
            print(f"Timestamp: {datetime.timedelta(seconds=ts)}\n")

        path, name, suffix = split_filename(self.filename)
        # rename last output file
        os.rename(self.temp_file, Path(f"{path}/{name}.processed.{suffix}"))


def main(args):
    if args.SP is None:
        pipe = Pipeline(
            args.file,
            args.language,
            args.bpe,
            args.replace_nums,
            args.max_lines
        )
        pipe.run()
    else:
        path, name, suffix = split_filename(args.file)
        modelname = f"{path}/model.sentencepiece.{args.SP}.{args.language}"
        # if model is already trained, load model
        if os.path.isfile(modelname):
            trained_model = False
            sp = spm.SentencePieceProcessor(
                model_file=modelname
            )

        else:
            # model needs to be trained
            trained_model = True
            sp_args = [
                f"--input={args.file}",
                f"--model_prefix={args.language}",
                f"--vocab_size={args.SP}",
                "--bos_id=-1",
                "--eos_id=-1"
            ]

            # add limit input sentence
            if args.max_lines != 0:
                sp_args.append(f"--input_sentence_size={args.max_lines}")

            # train and load model
            spm.SentencePieceTrainer.train(" ".join(sp_args))
            sp = spm.SentencePieceProcessor(
                model_file=f"{args.language}.model"
            )

        outputfile = Path(f"{path}/{name}.processed.{suffix}")

        # tokenize file
        with open(Path(args.file)) as infile:
            with open(outputfile, "w", encoding="utf-8") as ofile:
                for i, line in enumerate(infile):
                    encoded = sp.encode(line.strip(), out_type=str)
                    ofile.write(f"{' '.join(encoded)}\n")
                    print(f"Preprocessing: line {i:,}", end="\r")

        # if a new model was trained, move model and vocab to the directory
        # where the input file (and output file) are also saved
        if trained_model is True:
            os.rename(f"{args.language}.model", modelname)
            os.rename(
                f"{args.language}.vocab",
                f"{path}/vocab.{args.language}"
            )
    print(" " * 50, end="\r")
    print("Preprocessing: complete")
