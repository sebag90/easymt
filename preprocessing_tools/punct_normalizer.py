from sacremoses import MosesPunctNormalizer


class PunctNormalizer:
    def __init__(self, language):
        self.language = language
        self.normalizer = MosesPunctNormalizer(lang=language)

    def __repr__(self):
        return f"PunctNormalizer({self.language})"

    def __call__(self, line):
        return self.normalizer.normalize(line).strip()
