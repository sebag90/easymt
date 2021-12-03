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
        self.cpu = os.cpu_count() / 2

    def dexml(self):
        script = Path(f"preprocessing-tools/de-xml.perl")
        infile = Path(f"data/{self.name}")
        ofile = Path(f"data/dexmled")

        # execute command
        command = (
            f"perl {script} {infile} {self.l1} {self.l2} {ofile}"
        )
        os.system(command)

        # return files to be deleted
        return f"data/dexmled.{self.l1}", f"data/dexmled.{self.l2}"

    def clean(self):
        script = Path(f"preprocessing-tools/clean_corpus.perl")
        infile = Path(f"data/dexmled")
        ofile = Path(f"data/clean")

        # execute command
        command = (
            f"perl {script} {infile} {self.l1} {self.l2} {ofile} 1 {self.max_len}"
        )
        os.system(command)

        # return files to be deleted
        return f"data/clean.{self.l1}", f"data/clean.{self.l2}"

    def normalize(self, language):
        script = Path(f"preprocessing-tools/normalize-punctuation.perl")
        infile = Path(f"data/clean.{language}")
        ofile = Path(f"data/{self.name}.norm.{language}")

        # execute command
        command = (
            f"perl {script} -l {language} < {infile} > {ofile}"
        )
        os.system(command)

        # return files to be deleted
        return f"data/{self.name}.norm.{language}"

    def tokenize(self, language):
        script = Path(f"preprocessing-tools/tokenizer.perl")
        infile = Path(f"data/{self.name}.norm.{language}")
        ofile = Path(f"data/{self.name}.tok.{language}")

        # execute command
        command = (
            f"perl {script} -l {language} -no-escape -threads {self.cpu} "
            f"< {infile} > {ofile}"
        )
        os.system(command)

        # return files to be deleted
        return f"data/{self.name}.tok.{language}"

    def train_truecase(self, language):
        # create folder to store truecase model
        folder = Path("data/truecasing_models")
        os.makedirs(folder, exist_ok=True)

        # if model does not exists, train a new one
        model = Path(f"{folder}/model.{language}")
        if not os.path.isfile(model):
            script = Path(f"preprocessing-tools/train-truecaser.perl")
            infile = Path(f"data/{self.name}.tok.{language}")

            # execute command
            command = (
                f"perl {script} -corpus {infile} -model {model}"
            )
            os.system(command)

    def true_case(self, language):
        script = Path(f"preprocessing-tools/truecase.perl")
        model = Path(f"data/truecasing_models/model.{language}")
        infile = Path(f"data/{self.name}.tok.{language}")
        ofile = Path(f"data/{self.name}.true.{language}")

        # execute command
        command = (
            f"perl {script} -model {model} < {infile} > {ofile}"
        )
        os.system(command)

        # return files to be deleted
        return f"data/{self.name}.true.{language}"

    def learn_bpe(self, language):
        # create folder to store truecase model
        folder = Path("data/subword_models")
        os.makedirs(folder, exist_ok=True)

        # if model does not exists train a new one
        model = Path(f"{folder}/model.{self.bpe}.{language}")
        if not os.path.isfile(model):
            infile = Path(f"data/{self.name}.true.{language}")

            # execute command
            command = (
                f"subword-nmt learn-bpe -s {self.bpe} < {infile} > {model}"
            )
            os.system(command)

    def bpe_splitting(self, language):
        model = Path(f"data/subword_models/model.{self.bpe}.{language}")
        infile = Path(f"data/{self.name}.true.{language}")
        ofile = Path(f"data/{self.name}.bpe.{language}")

        # execute command
        command = (
            f"subword-nmt apply-bpe -c {model} < {infile} > {ofile}"
        )
        os.system(command)

        # return files to be deleted
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

    # MULTI
    if not args.single:
        # make sure files exist
        files = [
            Path(f"data/{config.dataset.name}.{config.dataset.source}"),
            Path(f"data/{config.dataset.name}.{config.dataset.target}")
        ]

        for filename in files:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Missing File: {filename}")

        # process 2 languages in parallel
        pipeline = PreprocessPipeline(
            config.dataset.name,
            config.dataset.source,
            config.dataset.target,
            config.dataset.subword_split,
            config.model.max_length
        )
        pipeline.multi()

    # SINGLE
    else:
        # make sure file exists
        if not os.path.isfile(args.single):
            raise FileNotFoundError(f"Missing File: {args.single}")
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
