import multiprocessing as mp
import os
from pathlib import Path
import re
from shutil import copyfile

from utils.parameters import Parameters


class PreprocessPipeline:

    def __init__(self, filename, l1, l2, bpe, max_len, single_file=False):
        self.name = filename
        self.l1 = l1
        self.l2 = l2
        self.bpe = bpe
        self.max_len = max_len
        self.single_file = single_file
        self.cpu = os.cpu_count()

    def dexml(self):
        command = (
            f"perl preprocessing-tools/de-xml.perl"
            f" data/{self.name} {self.l1} {self.l2} dexmled"
        )

        os.system(command)

        os.rename(f"dexmled.{self.l1}", f"data/dexmled.{self.l1}")
        os.rename(f"dexmled.{self.l2}", f"data/dexmled.{self.l2}")

        return f"data/dexmled.{self.l1}", f"data/dexmled.{self.l2}"

    def clean(self):
        command = (
            f"perl preprocessing-tools/clean_corpus.perl "
            f"data/dexmled {self.l1} {self.l2} clean 1 {self.max_len}"
        )
        os.system(command)
        os.rename(f"clean.{self.l1}", f"data/clean.{self.l1}")
        os.rename(f"clean.{self.l2}", f"data/clean.{self.l2}")
        return f"data/clean.{self.l1}", f"data/clean.{self.l2}"

    def normalize(self, language):
        command = (
            f"perl preprocessing-tools/normalize-punctuation.perl"
            f" -l {language} < data/clean.{language}"
            f" > data/{self.name}.norm.{language}"
        )

        os.system(command)
        return f"data/{self.name}.norm.{language}"

    def tokenize(self, language):
        command = (
            f"perl preprocessing-tools/tokenizer.perl -l {language}"
            f" -no-escape -threads {self.cpu} < "
            f"data/{self.name}.norm.{language}"
            f" > data/{self.name}.tok.{language}"
        )

        os.system(command)
        return f"data/{self.name}.tok.{language}"

    def train_truecase(self, language):
        # create folder to store truecase model
        folder = Path("data/truecasing_models")
        os.makedirs(folder, exist_ok=True)

        # if model does not exists, train a new one
        model = Path(f"{folder}/model.{language}")
        if not os.path.isfile(model):
            command = (
                f"perl preprocessing-tools/train-truecaser.perl "
                f"-corpus data/{self.name}.tok.{language}"
                f" -model {model}"
            )

            os.system(command)

    def true_case(self, language):
        command = (
            f"perl preprocessing-tools/truecase.perl -model "
            f"data/truecasing_models/model.{language}"
            f" <  data/{self.name}.tok.{language} "
            f"> data/{self.name}.true.{language}"
        )

        os.system(command)
        return f"data/{self.name}.true.{language}"

    def learn_bpe(self, language):
        # create folder to store truecase model
        folder = Path("data/subword_models")
        os.makedirs(folder, exist_ok=True)

        # if model does not exists train a new one
        model = Path(f"{folder}/model.{self.bpe}.{language}")
        if not os.path.isfile(model):
            command = (
                f"subword-nmt learn-bpe -s {self.bpe} "
                f"< data/{self.name}.true.{language} "
                f"> {model}"
            )

            os.system(command)

    def bpe_splitting(self, language):
        command = (
            f"subword-nmt apply-bpe -c data/subword_models/model.{self.bpe}.{language}"
            f" < data/{self.name}.true.{language} > "
            f"data/{self.name}.bpe.{language}")

        os.system(command)
        return f"data/{self.name}.bpe.{language}"

    def clean_join(self):
        to_remove = []

        # clean corpus
        print("Removing xml tags")
        l1, l2 = self.dexml()
        to_remove.append(l1)
        to_remove.append(l2)

        print("Cleaning corpus")
        l1, l2 = self.clean()
        to_remove.append(l1)
        to_remove.append(l2)

        return to_remove

    def single(self, language):
        to_remove = []
        print("Normalizing")
        to_remove.append(self.normalize(language))

        print("Tokenizing")
        to_remove.append(self.tokenize(language))

        print("Truecasing")
        to_remove.append(self.train_truecase(language))
        to_remove.append(self.true_case(language))

        if self.bpe > 0:
            print("Applying subword-splitting")
            to_remove.append(self.learn_bpe(language))
            to_remove.append(self.bpe_splitting(language))

        last = to_remove.pop()

        # rename last file
        os.rename(last, f"data/{self.name}_processed.{language}")

        return to_remove

    def multi(self):

        to_remove = self.clean_join()

        languages = [self.l1, self.l2]

        with mp.Pool() as pool:
            lists = pool.map(self.single, languages)

        for l in lists:
            to_remove += l

        print("Preprocessing complete")

        for file in to_remove:
            if file is not None:
                os.remove(file)


def preprocess(args):
    config = Parameters.from_config(args.path)

    if not args.single:
        # process 2 languages in parallel
        pipeline = PreprocessPipeline(
            config.dataset.name,
            config.dataset.source,
            config.dataset.target,
            config.dataset.subword_split,
            config.model.max_length
        )
        pipeline.multi()

    else:
        # process one single file
        name = args.single.split(os.sep)[-1]
        name = re.match(r"(.*)\.", name).group(1)

        copyfile(args.single, f"data/clean.{config.dataset.source}")
        pipeline = PreprocessPipeline(
            name,
            config.dataset.source,
            config.dataset.target,
            config.dataset.subword_split,
            config.model.max_length,
            single_file=True
        )
        to_remove = pipeline.single(config.dataset.source)
        to_remove.append(f"data/clean.{config.dataset.source}")
        for file in to_remove:
            if file is not None:
                os.remove(file)
