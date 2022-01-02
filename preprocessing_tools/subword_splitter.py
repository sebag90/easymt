import os
from pathlib import Path

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

from utils.errors import UntrainedModel


class SubwordSplitter:
    def __init__(self, language, bpe, path):
        self.language = language
        self.bpe = bpe
        self.model = Path(f"{path}/model.subword.{bpe}.{language}")
        if self.trained:
            with open(self.model, "r", encoding="utf-8") as modfile:
                self.splitter = BPE(modfile)

    def __repr__(self):
        return f"SubwordSplitter({self.language}, {self.bpe})"

    @property
    def trained(self):
        return os.path.isfile(self.model)

    def __call__(self, line):
        if os.path.isfile(self.model):
            return self.splitter.process_line(line).strip()
        else:
            raise UntrainedModel("SubwordSplitter not trained")

    def train(self, filename):
        if not os.path.isfile(self.model):
            print("Training subword model")

            # train and save model
            with open(filename, "r", encoding="utf-8") as infile, \
                    open(self.model, "w", encoding="utf-8") as ofile:
                learn_bpe(infile, ofile, self.bpe)

            # load model
            with open(self.model, "r", encoding="utf-8") as modfile:
                self.splitter = BPE(modfile)


if __name__ == "__main__":
    t = SubwordSplitter("en", 35000)
    t.train("data/test.en")
    print(t('catastrophically furiously'))
