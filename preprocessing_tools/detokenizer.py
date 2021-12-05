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
        result = subprocess.run(
            self.args, input=str.encode(f"{line}\n"),
            capture_output=True
        )

        return result.stdout.decode("UTF-8").strip()


if __name__ == "__main__":
    t = Detokenizer("en")
    print(t("Ciao . my name is ?"))
