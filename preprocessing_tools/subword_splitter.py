import subprocess
import os
from pathlib import Path


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

    @property
    def trained(self):
        return os.path.isfile(self.model)

    def __call__(self, line):
        if os.path.isfile(self.model):
            pipe = subprocess.Popen(
                ["echo", line],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )

            return subprocess.check_output(
                self.args,
                stdin=pipe.stdout
            ).decode("UTF-8")
        else:
            raise ValueError("Truecaser not trained")

    def train(self, filename):
        if not os.path.isfile(self.model):
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
