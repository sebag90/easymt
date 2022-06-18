import os
from pathlib import Path

from utils.errors import UntrainedModel

from sacremoses import MosesTruecaser


class Truecaser:
    def __init__(self, language, model=None):
        self.language = language
        if model is not None:
            self.truecaser = model
            self.trained = True
        else:
            self.truecaser = MosesTruecaser()
            self.trained = False

    def __repr__(self):
        return f"Truecaser({self.language})"

    def __call__(self, line):
        if self.trained is True:
            toks = self.truecaser.truecase(line)
            string = " ".join(toks)
            return string.strip()
        else:
            raise UntrainedModel("Truecaser not trained")

    def train(self, filename):
        if self.trained is False:
            self.truecaser.train_from_file_object(filename)
            self.trained = True



if __name__ == "__main__":
    t = Truecaser("en")
    t.train("data/train.en")
    print(t('ben austria france salzburg are countries'))
    print(t('ben austria france salzburg are countries'))
