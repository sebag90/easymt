import configparser
import datetime
import math
import os
from pathlib import Path
import time

import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import AttentionDecoder
from model.seq2seq import seq2seq
from model.loss import MaskedLoss

from utils.lang import Language
from utils.dataset import DataLoader


class Memory:
    """
    class to keep track of loss
    during training
    """
    def __init__(self):
        self.print_loss = 0
        self.epoch_loss = 0
        self.n_totals = 0
        self.epoch_totals = 0

    def print_reset(self):
        self.print_loss = 0
        self.n_totals = 0


class Parameters():
    """
    empty class to store parameters
    from the configuration file
    """
    def __init__(self):
        pass


class Trainer:
    def __init__(self, args):
        self.resume = args.resume
        self.steps = 0
        self.read_configuration(args.path)
        # pick device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        cpu = os.cpu_count()
        torch.set_num_threads(cpu)

    def read_configuration(self, path):
        # argument parser
        config = configparser.ConfigParser()
        config.read(path)

        # create empty class to store parameters
        params = Parameters()

        # read configuration
        params.filename = config["DATASET"]["name"]
        params.src_lang = config["DATASET"]["source"]
        params.tgt_lang = config["DATASET"]["target"]
        params.attention = config["MODEL"]["attention"]
        params.p_every = int(
            config["TRAINING"]["print_every"]
        )
        params.epochs = int(
            config["TRAINING"]["epochs"]
        )
        params.max_len = int(
            config["MODEL"]["max_length"]
        )
        params.batch_size = int(
            config["TRAINING"]["batch_size"]
        )
        params.bpe = int(
            config["DATASET"]["subword_split"]
        )
        params.save_every = int(
            config["TRAINING"]["save_every"]
        )
        params.teacher_forcing_ratio = float(
            config["TRAINING"]["teacher_ratio"]
        )
        params.hidden_size = int(
            config["MODEL"]["hidden_size"]
        )
        params.word_vec_size = int(
            config["MODEL"]["word_vec_size"]
        )
        params.l_rate = float(
            config["TRAINING"]["learning_rate"]
        )
        params.layers = int(
            config["MODEL"]["layers"]
        )
        params.bidirectional = eval(
            config["MODEL"]["bidirectional"]
        )
        params.dropout = float(
            config["MODEL"]["dropout"]
        )
        params.teacher = eval(
            config["TRAINING"]["teacher"]
        )
        params.valid_steps = int(
            config["TRAINING"]["valid_steps"]
        )

        self.params = params

    def read_data(self):
        if self.resume is None:
            # create language objects
            self.src_language = Language(self.params.src_lang)
            self.tgt_language = Language(self.params.tgt_lang)
            # read vocabulary from file
            self.src_language.read_vocabulary(
                Path(f"data/vocab.{self.src_language.name}")
            )
            self.tgt_language.read_vocabulary(
                Path(f"data/vocab.{self.tgt_language.name}")
            )

        else:
            # load from file
            self.model = seq2seq.load(Path(self.resume))
            self.src_language = self.model.src_lang
            self.tgt_language = self.model.tgt_lang

        self.train_data = DataLoader.from_files(
            "train", self.src_language, self.tgt_language,
            self.params.max_len, self.params.batch_size
        )

        self.eval_data = DataLoader.from_files(
            "eval", self.src_language, self.tgt_language,
            self.params.max_len, self.params.batch_size
        )

    def create_model(self):
        if self.resume is None:
            encoder = Encoder(
                self.src_language.n_words,
                self.params.word_vec_size,
                self.params.hidden_size,
                self.params.layers,
                self.params.dropout,
                self.params.bidirectional
            )
            decoder = AttentionDecoder(
                self.params.attention,
                self.params.word_vec_size,
                self.params.hidden_size,
                self.tgt_language.n_words,
                self.params.layers,
                self.params.dropout
            )

            self.model = seq2seq(
                encoder,
                decoder,
                self.src_language,
                self.tgt_language,
                self.params.max_len,
                self.params.bpe,
                epoch_trained=0,
                history=None
            )
        print(self.model, flush=True)
        # move model to device
        self.model.to(self.device)

        # set training mode
        self.model.train()

        # loss
        self.criterion = MaskedLoss()

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.l_rate
        )
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.params.l_rate
        # )

        # # scheduler:
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5
        )

    def save_model(self):
        # move model to cpu
        self.model.to("cpu")

        print("saving model...", flush=True)

        # save model
        os.makedirs("pretrained_models", exist_ok=True)
        self.model.save("pretrained_models/")

    @torch.no_grad()
    def evaluate(self):
        losses = []
        for batch in self.eval_data:
            loss, _ = self.model.train_batch(
                batch,
                Memory(),
                self.device,
                0,
                self.criterion
            )
            losses.append(loss)

        return torch.tensor(losses).mean()

    def train_loop(self):
        t_init = time.time()
        # test_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        for epoch in range(self.params.epochs):
            # initialize variables for monitoring
            memory = Memory()

            # shuffle data
            self.train_data.create_order()

            # add history element
            self.model.history[self.model.epoch_trained + 1] = list()

            # start training loop over batches
            for i, batch in enumerate(self.train_data):
                self.optimizer.zero_grad()

                # process batch
                loss, memory = self.model.train_batch(
                    batch,
                    memory,
                    self.device,
                    self.params.teacher_forcing_ratio,
                    self.criterion
                )

                # calculate gradient
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                # optimizer step
                self.optimizer.step()
                self.steps += 1

                # validation step
                if self.steps % self.params.valid_steps == 0:
                    eval_loss = self.evaluate()
                    self.scheduler.step(eval_loss)
                    print("-"*len(to_print), flush=True)
                    print(f"Validation loss: {round((eval_loss.item()), 5):.5f}")
                    print("-"*len(to_print), flush=True)

                # print every x batches
                if i % self.params.p_every == 0:
                    t_1 = time.time()
                    ts = int(t_1 - t_init)
                    print_loss = memory.print_loss / memory.n_totals
                    ppl = math.exp(print_loss)
                    lr = self.optimizer.param_groups[0]['lr']
                    print_time = datetime.timedelta(seconds=ts)
                    to_print = (
                        f"Epoch: {epoch + 1}/{self.params.epochs} "
                        f"[{i + 1}/{len(self.train_data) + 1}] | "
                        f"lr: {lr} | "
                        f"Loss: {round((print_loss), 5):.5f} | "
                        f"ppl: {round(ppl, 5):.5f} | "
                        f"Time: {print_time}"
                    )

                    print(to_print, flush=True)

                    # add loss to history
                    idx = self.model.epoch_trained + 1
                    self.model.history[idx].append(print_loss)

                    # reset loss
                    memory.print_reset()

            # calculate epoch loss
            print("-"*len(to_print), flush=True)
            epoch_loss = round(
                (memory.epoch_loss / memory.epoch_totals), 5
            )

            print(f"Epoch loss:\t{epoch_loss}", flush=True)
            self.model.epoch_trained += 1
            if self.params.epochs != 1 and epoch + 1 != self.params.epochs:
                if self.params.save_every != 0:
                    if epoch % (self.params.save_every - 1) == 0:
                        self.save_model()
                        self.model.to(self.device)
            print("-"*len(to_print), flush=True)

        # calculate and print total time for training
        t_end = time.time()
        ts = int(t_end - t_init)
        print_time = datetime.timedelta(seconds=ts)
        print(f"Training completed in: {print_time}", flush=True)


def train(args):
    trainer = Trainer(args)
    trainer.read_data()
    trainer.create_model()
    try:
        trainer.train_loop()
    except KeyboardInterrupt:
        pass
    trainer.save_model()
