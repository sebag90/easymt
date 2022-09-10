import sys
import tempfile
import re

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

from utils.errors import UntrainedModel


class SubwordSplitter:
    def __init__(self, language, bpe):
        self.language = language
        self.bpe = bpe
        self.trained = False
        self.pattern = re.compile(r"@@( |$)")

    def __repr__(self):
        return f"SubwordSplitter({self.language}, {self.bpe})"

    def __call__(self, line):
        if self.trained is True:
            return self.model.process_line(line).strip()
        else:
            raise UntrainedModel("SubwordSplitter not trained")

    def train(self, filename):
        if self.trained is False:
            print("Training subword model", file=sys.stderr)
            modfile = tempfile.TemporaryFile("w+")
            filename.seek(0)

            learn_bpe(filename, modfile, self.bpe)
            self.model = BPE(modfile)
            modfile.close()
            self.trained = True

    def decode(self, line):
        return re.sub(self.pattern, "", line)
