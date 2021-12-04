import subprocess


class Detokenizer:
    def __init__(self, language):
        self.language = language
        self.args = [
            "perl",
            "preprocessing_tools/perl_scripts/detokenizer.perl",
            "-q",
            "-u",
            "-l",
            language
        ]

    def __repr__(self):
        return f"Detokenizer({self.language})"

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
    t = Detokenizer("en")
    print(t("Ciao . my name is ?"))
