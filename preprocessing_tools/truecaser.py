import subprocess
import os
from pathlib import Path

from utils.errors import UntrainedModel


class Truecaser:
    def __init__(self, language):
        self.language = language
        self.model = Path(f"data/truecasing_models/model.{language}")
        self.args = [
            "perl",
            "preprocessing_tools/perl_scripts/truecase.perl",
            "--model",
            self.model
        ]

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
            raise UntrainedModel("Truecaser not trained")

    def train(self, filename):
        if not os.path.isfile(self.model):
            script = Path(f"perl_scripts/train-truecaser.perl")
            print("Training truecasing model")
            # execute command
            command = (
                f"perl {script} -corpus {filename} -model {self.model}"
            )
            os.system(command)


if __name__ == "__main__":
    t = Truecaser("en")
    t.train("../data/train.en")
    print(t('And this is it'))
