import os
from pathlib import Path

from utils.errors import UntrainedModel

from sacremoses import MosesTruecaser


class Truecaser:
    def __init__(self, language, path):
        self.language = language
        self.model = Path(f"{path}/model.truecase.{language}")

        if self.trained:
            self.truecaser = MosesTruecaser(self.model)
        else:
            self.truecaser = MosesTruecaser()

    def __repr__(self):
        return f"Truecaser({self.language})"

    @property
    def trained(self):
        return os.path.isfile(self.model)

    def __call__(self, line):
        if os.path.isfile(self.model):
            toks = self.truecaser.truecase(line)
            string = " ".join(toks)
            return string.strip()
        else:
            raise UntrainedModel("Truecaser not trained")

    def train(self, filename):
        if not os.path.isfile(self.model):
            self.truecaser.train_from_file(
                filename, save_to=self.model
            )


if __name__ == "__main__":
    t = Truecaser("en")
    t.train("data/train.en")
    print(t('ben austria france salzburg are countries'))
    print(t('ben austria france salzburg are countries'))
