import subprocess
import os
from pathlib import Path

from utils.errors import UntrainedModel


class SubwordSplitter:
    def __init__(self, language, bpe):
        self.language = language
        self.bpe = bpe
        self.model = Path(f"data/subword_models/model.{bpe}.{language}")
        self.args = [
            "subword-nmt",
            "apply-bpe",
            "-c",
            self.model
        ]

    def __repr__(self):
        return f"SubwordSplitter({self.language}, {self.bpe})"

    @property
    def trained(self):
        return os.path.isfile(self.model)

    def __call__(self, line):
        if os.path.isfile(self.model):
            result = subprocess.run(
                self.args, input=str.encode(f"{line}\n"),
                capture_output=True
            )

            return result.stdout.decode("UTF-8").strip()
        else:
            raise UntrainedModel("SubwordSplitter not trained")

    def train(self, filename):
        if not os.path.isfile(self.model):
            os.makedirs(Path("data/subword_models"), exist_ok=True)
            print("Training subword model")
            # execute command
            command = (
                f"subword-nmt learn-bpe -s {self.bpe} "
                f"< {filename} "
                f"> {self.model}"
            )
            os.system(command)


if __name__ == "__main__":
    t = SubwordSplitter("en", 35000)
    t.train("data/test.en")
    print(t('catastrophically furiously'))
