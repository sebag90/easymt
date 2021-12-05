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

    def __repr__(self):
        return f"Truecaser({self.language})"

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
            raise UntrainedModel("Truecaser not trained")

    def train(self, filename):
        if not os.path.isfile(self.model):
            os.makedirs(Path("data/truecasing_models"), exist_ok=True)
            script = Path(
                f"preprocessing_tools/perl_scripts/train-truecaser.perl"
            )
            print("Training truecasing model")
            # execute command
            command = (
                f"perl {script} -corpus {filename} -model {self.model}"
            )
            os.system(command)


if __name__ == "__main__":
    t = Truecaser("en")
    t.train("../data/train.en")
    print(t('Right-wing populists triumph in Austria , have total of 29 percent '))
