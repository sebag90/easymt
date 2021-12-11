import subprocess


class Detokenizer:
    def __init__(self, language):
        self.language = language
        self.args = [
            "perl",
            "preprocessing_tools/perl_scripts/detokenizer.perl",
            "-q",
            "-u",
            "-b",
            "-l",
            language
        ]
        self.proc = subprocess.Popen(
            self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

    def __repr__(self):
        return f"Detokenizer({self.language})"

    def __call__(self, line):
        self.proc.stdin.write(str.encode(f"{line}\n"))
        self.proc.stdin.flush()
        return self.proc.stdout.readline().decode("UTF-8").strip()


if __name__ == "__main__":
    t = Detokenizer("en")
    print(t("Ciao . my name is ?"))
    print(t("Ciao . my name is ?"))
