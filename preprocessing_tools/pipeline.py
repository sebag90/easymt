import sys
import time
import tempfile
import datetime

from preprocessing_tools.tokenizer import Tokenizer
from preprocessing_tools.truecaser import Truecaser
from preprocessing_tools.punct_normalizer import PunctNormalizer
from preprocessing_tools.subword_splitter import SubwordSplitter
from preprocessing_tools.num_replacer import NumReplacer
from preprocessing_tools.lower_caser import LowerCaser


class Pipeline:
    def __init__(self, language, bpe, remove_nums, max_lines):
        self.max_lines = max_lines
        self.language = language

        self.pipe = [
            PunctNormalizer(language),
            Tokenizer(language)
        ]

        if remove_nums is True:
            self.pipe.append(
                NumReplacer()
            )

        tr = Truecaser(language, None)
        lc = LowerCaser(tr.trained)

        self.trainable = [
            tr,
            lc  # must be applied after training Truecaser
        ]

        # add bpe splitter to trainable processors
        if bpe is not None:
            self.trainable.append(
                SubwordSplitter(language, bpe)
            )

    def apply_trainable(self, processor, temp_file):
        train_file = tempfile.TemporaryFile("w+")
        # train processor on last produced text file
        if self.max_lines == 0:
            processor.train(temp_file)
        else:
            for i, line in enumerate(temp_file):
                train_file.write(line)
                if i == self.max_lines:
                    break

            # train on reduced file
            processor.train(train_file)
            train_file.close()

        # do not apply truecaser
        if not isinstance(processor, Truecaser):
            stepfile = tempfile.TemporaryFile("w+")
            temp_file.seek(0)
            for i, line in enumerate(temp_file):
                line = processor(line)
                stepfile.write(f"{line}\n")
                if (i+1) % 100000 == 0:
                    print(f"Processed lines: {i + 1:,}", file=sys.stderr)

            temp_file.close()
            return stepfile
        else:
            return temp_file

    def run(self, input_gen):
        to_train = list()
        # collect trained processors
        for processor in self.trainable:
            if processor.trained:
                self.pipe.append(processor)
            else:
                to_train.append(processor)

        pipe_string = " > ".join([str(i) for i in self.pipe])
        complete = " > ".join([str(i) for i in self.pipe + to_train])
        print(f"Pipe: {complete}\n", file=sys.stderr)
        t_0 = time.time()

        print(f"Applying: {pipe_string}", file=sys.stderr)
        # applied untrainable and trained processors

        t_file = tempfile.TemporaryFile("w+")
        for i, line in enumerate(input_gen):
            for processor in self.pipe:
                if not isinstance(processor, Truecaser):
                    line = processor(line)

            t_file.write(f"{line}\n")
            if (i+1) % 100000 == 0:
                print(f"Processed lines: {i + 1:,}", file=sys.stderr)

        t_1 = time.time()
        ts = int(t_1 - t_0)
        print(f"Timestamp: {datetime.timedelta(seconds=ts)}\n", file=sys.stderr)

        # train and apply untrained processors
        for processor in to_train:
            print(f"Applying: {processor}", file=sys.stderr)
            t_file = self.apply_trainable(processor, t_file)

            t_1 = time.time()
            ts = int(t_1 - t_0)
            print(f"Timestamp: {datetime.timedelta(seconds=ts)}\n", file=sys.stderr)

        t_file.seek(0)
        for line in t_file:
            sys.stdout.write(line)

        t_file.close()