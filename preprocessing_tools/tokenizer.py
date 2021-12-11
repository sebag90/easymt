from sacremoses import MosesTokenizer


class Tokenizer:
    def __init__(self, language):
        self.language = language
        self.tokenizer = MosesTokenizer(lang=language)

    def __repr__(self):
        return f"Tokenizer({self.language})"

    def __call__(self, line):
        return " ".join(self.tokenizer.tokenize(line, escape=False))
