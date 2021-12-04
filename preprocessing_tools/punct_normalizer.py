import subprocess


class PunctNormalizer:
    def __init__(self, language):
        self.language = language
        self.args = [
            "perl",
            "preprocessing_tools/perl_scripts/normalize-punctuation.perl",
            "-l",
            language
        ]

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
    t = PunctNormalizer("de")
    print(t('!  "sco", ----- "sco,",  !'))
