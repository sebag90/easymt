class LowerCaser:
    def __init__(self, trained=False):
        self.trained = trained

    def __repr__(self):
        return "Lowercaser"

    def __call__(self, line):
        return line.lower().strip()

    def train(self, path):
        pass