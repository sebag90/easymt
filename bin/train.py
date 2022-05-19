"""
train a model from scratch or resume training
"""

import datetime
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch
import torch.nn as nn

from model.model_generator import ModelGenerator
from model.loss import MaskedLoss
from model.optimizers import get_optimizer

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
        self._print_counter = 0

    @property
    def print_loss(self):
        return self._print_loss / self._print_counter

    def add(self, loss):
        self._print_loss += loss
        self._print_counter += 1

    def print_reset(self):
        self._print_loss = 0
        self._print_counter = 0


class Trainer:
    def __init__(self, resume, batched, mixed, params):
        self.resume = resume
        self.batched = batched
        self.params = params
        self.mixed = mixed
        self.scaler = torch.cuda.amp.GradScaler() if mixed is True else None

        # pick device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        torch.set_num_threads(os.cpu_count())

        # avoid abrpt termination of training by
        # calling the kill_training method to
        # ensure the last model is saved
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.kill_training)

    def kill_training(self, *args):
        print("Aborting...")
        self.optimizer.step()
        self.save_model()
        sys.exit()

    def read_data(self):
        """
        read and prepare train and eval dataset
        """
        if self.resume is None:
            # create language objects
            self.src_language = Language(self.params.model.source)
            self.tgt_language = Language(self.params.model.target)
            # read vocabulary from file
            self.src_language.read_vocabulary(
                Path(self.params.data.src_vocab)
            )
            self.tgt_language.read_vocabulary(
                Path(self.params.data.tgt_vocab)
            )

        else:
            # load from file
            self.checkpoint = torch.load(
                Path(self.resume),
                map_location=self.device
            )
            self.model = self.checkpoint["model"]

        # load eval dataset
        self.eval_data = DataLoader.from_files(
            self.params.data.src_eval,
            self.params.data.tgt_eval,
            self.params.model.max_length,
            self.params.training.batch_size
        )

        # load train dataset
        if self.batched:
            self.train_data = BatchedData(
                Path(self.batched),
                self.params.model.max_length,
                self.params.training.batch_size
                )

        else:
            self.train_data = DataLoader.from_files(
                self.params.data.src_train,
                self.params.data.tgt_train,
                self.params.model.max_length,
                self.params.training.batch_size
            )

    def create_model(self):
        """
        create a model, either from scratch or load it from file
        """
        if self.resume is None:
            model_generator = ModelGenerator()
            self.model = model_generator.generate_model(
                self.params,
                self.src_language,
                self.tgt_language
            )

            for param in self.model.parameters():
                if self.model.type == "rnn":
                    nn.init.uniform_(
                        param,
                        -self.params.model.uniform_init,
                        self.params.model.uniform_init
                    )
                else:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)

        print(self.model, flush=True)
        # move model to device
        self.model.to(self.device)

        # set training mode
        self.model.train()

        # loss
        self.criterion = MaskedLoss(
            padding_idx=0,
            smoothing=self.params.training.label_smoothing
        )

        self.optimizer = get_optimizer(self.model, self.params)

        # load checkpoint
        if self.resume:
            self.optimizer.load_state_dict(
                self.checkpoint["optimizer"]
            )

            # remove checkpoint attribute from trainer
            delattr(self, "checkpoint")

    def save_model(self):
        # save model
        os.makedirs("checkpoints", exist_ok=True)

        l1 = self.model.src_lang.name
        l2 = self.model.tgt_lang.name
        st = self.model.steps
        path = Path(f"checkpoints/{self.model.type}_{l1}-{l2}_{st}.pt")

        torch.save({
                "model": self.model,
                "optimizer": self.optimizer.state_dict()
            },
            path
        )
        print("Model saved", flush=True)

    @torch.no_grad()
    def evaluate(self):
        """
        evaluate the model on the evaluation data set
        """
        self.model.eval()

        losses = list()
        for batch in self.eval_data:
            loss = self.model(
                batch,
                self.device,
                1,  # with teacher for consistent results
                self.criterion
            )

            losses.append(loss)

        self.model.train()
        return torch.tensor(losses).mean()

    def train_loop(self):
        """
        main function of the train loop
        """
        # initialize helping parameters
        training = True
        steps = 0
        accumulation = (self.params.training.step_size !=
                        self.params.training.batch_size)
        acc_steps = 0
        sub_step = 0

        # start timer and training
        self.model.to(self.device)
        t_init = time.time()

        while training:
            # initialize variables for monitoring
            loss_memory = Memory()

            # shuffle data
            self.train_data.shuffle()

            # start training loop over batches
            for batch in self.train_data:
                batch_size = len(batch[0])
                acc_steps += batch_size
                # process batch
                if self.mixed is True:
                    with torch.cuda.amp.autocast():
                        loss = self.model(
                            batch,
                            self.device,
                            self.params.training.teacher_ratio,
                            self.criterion
                        )
                else:
                    loss = self.model(
                            batch,
                            self.device,
                            self.params.training.teacher_ratio,
                            self.criterion
                        )

                loss_memory.add(loss.item())
                if accumulation:
                    # scale loss if using gradient accumulation
                    norm = self.params.training.step_size / batch_size
                    # TODO: should also be in autocast env?
                    loss = loss / norm

                # calculate gradient
                if self.mixed is True:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer.optimizer)
                else:
                    loss.backward()

                if (not accumulation
                        or (acc_steps >= self.params.training.step_size)):
                    # gradient clipping
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.params.training.gradient_clipping
                    )

                    # optimizer step
                    self.optimizer.step(self.scaler)
                    self.model.steps += 1
                    steps += 1
                    acc_steps = 0

                if sub_step != steps:
                    sub_step = steps
                    # print every x steps
                    if (steps % self.params.training.print_every == 0
                            and steps != 0):
                        t_1 = time.time()
                        ts = int(t_1 - t_init)
                        print_loss = loss_memory.print_loss
                        ppl = math.exp(print_loss)
                        lr = self.optimizer.lr
                        print_time = datetime.timedelta(seconds=ts)
                        print_step = f"{steps}/{self.params.training.steps}"
                        to_print = (
                            f"step: {print_step:13} | "
                            f"lr: {round(lr, 5):7} | "
                            f"loss: {round((print_loss), 5):8.5f} | "
                            f"ppl: {round(ppl, 2):8.2f} | "
                            f"time: {print_time}"
                        )

                        print(to_print, flush=True)

                        # reset loss
                        loss_memory.print_reset()

                    # validation step
                    if (steps % self.params.training.valid_steps == 0
                            and steps != 0):
                        eval_loss = self.evaluate()
                        self.optimizer.scheduler_step(eval_loss)
                        delim = "-" * len(to_print)
                        val_loss = round((eval_loss.item()), 5)
                        print(
                            f"{delim}\n"
                            f"Validation loss: {val_loss:.5f}"
                            f"\n{delim}",
                            flush=True
                        )

                    # save model
                    if self.params.training.save_every != 0:
                        if (steps % self.params.training.save_every == 0
                                and steps != 0):
                            self.save_model()

                # check if end of training
                if steps == self.params.training.steps:
                    self.optimizer.step()
                    training = False
                    break

        # calculate and print total time for training
        t_end = time.time()
        ts = int(t_end - t_init)
        print_time = datetime.timedelta(seconds=ts)
        print(f"Training completed in: {print_time}", flush=True)


def main(args):
    # extract parameters
    params = Parameters.from_config(args.path)

    # initialize trainer
    trainer = Trainer(args.resume, args.batched, args.mixed, params)
    trainer.read_data()
    trainer.create_model()
    trainer.train_loop()
    trainer.save_model()
