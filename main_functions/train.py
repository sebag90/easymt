import datetime
import math
import os
from pathlib import Path
import time

import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import seq2seq
from model.loss import MaskedLoss

from utils.lang import Language
from utils.dataset import DataLoader, BatchedData
from utils.parameters import Parameters


class Memory:
    """
    class to keep track of loss
    during training
    """
    def __init__(self):
        self._print_loss = 0
        self._epoch_loss = 0
        self._epoch_counter = 0
        self._print_counter = 0

    @property
    def print_loss(self):
        return self._print_loss / self._print_counter

    @property
    def epoch_loss(self):
        return self._epoch_loss / self._epoch_counter

    def add(self, loss):
        self._print_loss += loss
        self._epoch_loss += loss
        self._print_counter += 1
        self._epoch_counter += 1

    def print_reset(self):
        self._print_loss = 0
        self._print_counter = 0


class Trainer:
    def __init__(self, args):
        self.resume = args.resume
        self.batched = args.batched
        self.steps = 0
        self.params = Parameters.from_config(args.path)

        # pick device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        cpu = os.cpu_count()
        torch.set_num_threads(cpu)

    def read_data(self):
        """
        read and prepare train and eval dataset
        """
        if self.resume is None:
            # create language objects
            self.src_language = Language(self.params.dataset.source)
            self.tgt_language = Language(self.params.dataset.target)
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

        # load eval dataset
        self.eval_data = DataLoader.from_files(
            "eval", self.src_language, self.tgt_language,
            self.params.model.max_length, self.params.training.batch_size
        )

        # load train dataset
        if self.batched:
            self.train_data = BatchedData(Path("data/batched"))

        else:
            self.train_data = DataLoader.from_files(
                "train", self.src_language, self.tgt_language,
                self.params.model.max_length, self.params.training.batch_size
            )

    def create_model(self):
        """
        create a model, either from scratch or load it from file
        """
        if self.resume is None:
            encoder = Encoder(
                self.src_language.n_words,
                self.params.model.word_vec_size,
                self.params.model.hidden_size,
                self.params.model.layers,
                self.params.model.dropout,
                self.params.model.bidirectional
            )
            decoder = Decoder(
                self.params.model.attention,
                self.params.model.word_vec_size,
                self.params.model.hidden_size,
                self.tgt_language.n_words,
                self.params.model.layers,
                self.params.model.dropout,
                self.params.model.input_feed
            )

            self.model = seq2seq(
                encoder,
                decoder,
                self.src_language,
                self.tgt_language,
                self.params.model.max_length,
                self.params.dataset.subword_split,
                epoch_trained=0,
                history=None
            )

            # initialize parameters uniformly
            for p in self.model.parameters():
                p.data.uniform_(
                    - self.params.model.uniform_init,
                    self.params.model.uniform_init
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
            lr=self.params.training.learning_rate
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
        """
        move model to cpu and save it
        """
        # move model to cpu
        self.model.to("cpu")

        print("saving model...", flush=True)

        # save model
        os.makedirs("pretrained_models", exist_ok=True)
        self.model.save("pretrained_models/")

    @torch.no_grad()
    def evaluate(self):
        """
        evaluate the model on the evaluation data set
        """
        self.model.eval()

        losses = list()
        for batch in self.eval_data:
            loss = self.model.train_batch(
                batch,
                self.device,
                0,
                self.criterion
            )

            losses.append(loss)

        self.model.train()
        return torch.tensor(losses).mean()

    def train_loop(self):
        """
        main function of the train loop
        """
        t_init = time.time()
        for epoch in range(self.params.training.epochs):
            # initialize variables for monitoring
            loss_memory = Memory()

            # shuffle data
            self.train_data.create_order()

            # add history element
            self.model.history[self.model.epoch_trained + 1] = list()

            # start training loop over batches
            for i, batch in enumerate(self.train_data):
                self.optimizer.zero_grad()

                # process batch
                loss = self.model.train_batch(
                    batch,
                    self.device,
                    self.params.training.teacher_ratio,
                    self.criterion
                )
                loss_memory.add(loss.item())

                # calculate gradient
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                # optimizer step
                self.optimizer.step()
                self.steps += 1

                # print every x batches
                if (i % self.params.training.print_every == 0
                        and i != 0):
                    t_1 = time.time()
                    ts = int(t_1 - t_init)
                    print_loss = loss_memory.print_loss
                    ppl = math.exp(print_loss)
                    lr = self.optimizer.param_groups[0]['lr']
                    print_time = datetime.timedelta(seconds=ts)
                    to_print = (
                        f"Epoch: {epoch + 1}/{self.params.training.epochs} "
                        f"[{i}/{len(self.train_data)}] | "
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
                    loss_memory.print_reset()

                # validation step
                if self.steps % self.params.training.valid_steps == 0:
                    eval_loss = self.evaluate()
                    self.scheduler.step(eval_loss)
                    print("-"*len(to_print), flush=True)
                    print(
                        f"Validation loss: {round((eval_loss.item()), 5):.5f}",
                        flush=True
                    )
                    print("-"*len(to_print), flush=True)

            # calculate epoch loss
            print("-"*len(to_print), flush=True)
            epoch_loss = round(
                loss_memory.epoch_loss, 5
            )

            print(f"Epoch loss:\t{epoch_loss}", flush=True)
            self.model.epoch_trained += 1
            if (self.params.training.epochs != 1 and
                    epoch + 1 != self.params.training.epochs):
                if self.params.training.save_every != 0:
                    if epoch % (self.params.training.save_every - 1) == 0:
                        # important! move back to GPU after saving
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
