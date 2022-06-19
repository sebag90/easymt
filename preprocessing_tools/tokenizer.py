from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer


class Tokenizer:
    def __init__(self, language):
        self.language = language
        self.tokenizer = MosesTokenizer(lang=language)
        self.detokenizer = MosesDetokenizer(lang=language)

    def __repr__(self):
        return f"Tokenizer({self.language})"

    def __call__(self, line):
        return " ".join(self.tokenizer.tokenize(line, escape=False))

    def decode(self, line):
        return self.detokenizer.detokenize(line.split())
