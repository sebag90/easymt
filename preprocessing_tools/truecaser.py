from sacremoses import MosesTruecaser

from utils.errors import UntrainedModel


class Truecaser:
    def __init__(self, language):
        self.language = language
        self.truecaser = MosesTruecaser()
        self.trained = False

    def __repr__(self):
        return f"Truecaser({self.language})"

    def __call__(self, line):
        return line

    def decode(self, line):
        if self.trained is True:
            toks = self.truecaser.truecase(line)
            string = " ".join(toks)
            return string.strip()
        else:
            raise UntrainedModel("Truecaser not trained")

    def train(self, filename):
        if self.trained is False:
            filename.seek(0)
            self.truecaser.train_from_file_object(filename)
            self.trained = True
