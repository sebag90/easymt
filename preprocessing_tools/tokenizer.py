import subprocess
import os


class Tokenizer:
    def __init__(self, language):
        self.language = language
        self.args = [
            "perl",
            "preprocessing_tools/perl_scripts/tokenizer.perl",
            "-q",
            "-l",
            language,
            "-no-escape",
            "-threads",
            str(int(os.cpu_count() / 2))
        ]

    def __repr__(self):
        return f"Tokenizer({self.language})"

    def __call__(self, line):
        result = subprocess.run(
            self.args, input=str.encode(f"{line}\n"),
            capture_output=True
        )

        return result.stdout.decode("UTF-8").strip()


if __name__ == "__main__":
    t = Tokenizer("en")
    print(t("Ciao. my name is?"))
