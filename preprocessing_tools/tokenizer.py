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
        pipe = subprocess.Popen(
            ["echo", line],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        return subprocess.check_output(
            self.args,
            stdin=pipe.stdout
        ).decode("UTF-8")


if __name__ == "__main__":
    t = Tokenizer("en")
    print(t("Ciao. my name is?"))
