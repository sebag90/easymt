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

    def __repr__(self):
        return f"PunctNormalizer({self.language})"

    def __call__(self, line):
        result = subprocess.run(
            self.args, input=str.encode(line),
            capture_output=True
        )

        return result.stdout.decode("UTF-8").strip()


if __name__ == "__main__":
    t = PunctNormalizer("de")
    print(t('!  "sco", ----- "sco,",  !'))
