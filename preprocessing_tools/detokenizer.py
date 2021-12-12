from sacremoses import MosesDetokenizer


class Detokenizer:
    def __init__(self, language):
        self.language = language
        self.detokenizer = MosesDetokenizer(lang=language)

    def __repr__(self):
        return f"Detokenizer({self.language})"

    def __call__(self, line):
        return self.detokenizer.detokenize(line.split())


if __name__ == "__main__":
    t = Detokenizer("en")
    print(t("Ciao . my name is ?"))
