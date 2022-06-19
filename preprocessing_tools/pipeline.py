import sys
import tempfile

from preprocessing_tools.tokenizer import Tokenizer
from preprocessing_tools.truecaser import Truecaser
from preprocessing_tools.punct_normalizer import PunctNormalizer
from preprocessing_tools.subword_splitter import SubwordSplitter
from preprocessing_tools.num_replacer import NumReplacer
from preprocessing_tools.lower_caser import LowerCaser
from preprocessing_tools.sentence_piece import SentencePieceTokenizer


class Tools:
    def __init__(self, language, bpe, sp_voc_size, max_lines):
        self.sp = SentencePieceTokenizer(
            sp_voc_size, max_lines
        )
        self.truecaser = Truecaser(language)
        self.punct_normalizer = PunctNormalizer(language)
        self.tokenizer = Tokenizer(language)
        self.num_replacer = NumReplacer()
        self.lower_caser = LowerCaser()
        self.subword_splitter = SubwordSplitter(language, bpe)


class Pipeline:
    def __init__(self, language, bpe, sp_voc_size, remove_nums, max_lines):
        self.max_lines = max_lines
        self.language = language
        self.tools = Tools(
            language, bpe, sp_voc_size, max_lines
        )
        self.create_pipe(bpe, sp_voc_size, remove_nums)

    def create_pipe(self, bpe, sp_voc_size, remove_nums):
        if sp_voc_size > 0:
            self.to_train = [
                "sp"
            ]
            self.pipe = list()
            self.decoder = ["sp"]

        else:
            self.bpe = bpe
            self.pipe = [
                "punct_normalizer",
                "tokenizer"
            ]
            self.decoder = [
                "truecaser", "tokenizer"
            ]

            if remove_nums is True:
                self.pipe.append(
                    "num_replacer"
                )

            self.to_train = [
                "truecaser",
                "lower_caser"
            ]

            if bpe > 0:
                self.to_train.append(
                    "subword_splitter"
                )
                self.decoder.insert(
                    0, "subword_splitter"
                )

    def get_model(self):
        return {
            "pipe": self.pipe,
            "decoder": self.decoder,
            "language": self.language,
            "tools": self.tools
        }

    @classmethod
    def from_trained_model(cls, model_dict):
        p = cls("placeholder", 0, 0, False, 0)
        p.pipe = model_dict["pipe"]
        p.decoder = model_dict["decoder"]
        p.language = model_dict["language"]
        p.tools = model_dict["tools"]
        p.to_train = list()
        return p

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

        stepfile = tempfile.TemporaryFile("w+")
        temp_file.seek(0)
        for i, line in enumerate(temp_file):
            line = processor(line)
            stepfile.write(f"{line}\n")
            if (i+1) % 100000 == 0:
                print(f"Processed lines: {i + 1:,}", file=sys.stderr)

        temp_file.close()
        return stepfile

    def run(self, input_gen):
        pipe_string = " > ".join(
            [str(getattr(self.tools, i)) for i in self.pipe]
        )
        complete = " > ".join(
            [str(getattr(self.tools, i)) for i in self.pipe + self.to_train]
        )
        print(f"Pipe: {complete}\n", file=sys.stderr)

        if len(self.pipe) > 0:
            print(f"Applying: {pipe_string}", file=sys.stderr)

        t_file = tempfile.TemporaryFile("w+")
        for i, line in enumerate(input_gen):
            for p_name in self.pipe:
                processor = getattr(self.tools, p_name)
                line = processor(line)

            t_file.write(f"{line}\n")

            if len(self.pipe) > 0:
                if (i+1) % 100000 == 0:
                    print(f"Processed lines: {i + 1:,}", file=sys.stderr)

        # train and apply untrained processors
        for p_name in self.to_train:
            processor = getattr(self.tools, p_name)
            print(f"Applying: {processor}", file=sys.stderr)
            t_file = self.apply_trainable(processor, t_file)
            self.pipe.append(p_name)

        t_file.seek(0)
        for line in t_file:
            sys.stdout.write(line)

        t_file.close()

    def decode(self, line):
        for p_name in self.decoder:
            processor = getattr(self.tools, p_name)
            line = processor.decode(line)

        return line
