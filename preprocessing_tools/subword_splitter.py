import os
import sys
import tempfile

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

from utils.errors import UntrainedModel


class SubwordSplitter:
    def __init__(self, language, bpe, model=None):
        self.language = language
        self.bpe = bpe
        if model is None:
            self.trained = False
        else:
            self.trained = True
            self.model = model

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
            self.trained = True


if __name__ == "__main__":
    import requests
    import random
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"

    response = requests.get(word_site)
    WORDS = response.content.splitlines()

    

    t = SubwordSplitter("en", 1000)
    n = tempfile.TemporaryFile("w+")
    
    for i in range(100):
        for w in range(250):
            n.write(str(random.choice(WORDS)))


    t.train(n)
    print(t('catastrophically furiously'))
