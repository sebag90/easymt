import torch


class Optimizer:
    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self, loss):
        self.scheduler.step(loss)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']


class NoamOpt(Optimizer):
    """
    Optim wrapper that implements rate.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        super().__init__(optimizer, None)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def scheduler_step(self, loss):
        """
        integrated scheduler
        """
        pass

    def rate(self, step=None):
        """
        calculate learning rate
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) * min(
                step ** (-0.5), step * self.warmup ** (-1.5)
                )
            )


def get_optimizer(model, params):
    if params.training.optimizer == "noam":
        opt = NoamOpt(
            params.transformer.d_model,
            params.transformer.noam_factor,
            params.transformer.warm_up,
            torch.optim.Adam(
                model.parameters(),
                lr=0,
                betas=(0.9, 0.98),
                eps=1e-9)
            )

        return opt

    # optimizer
    if params.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.training.learning_rate
        )

    elif params.training.optimizer.upper() == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params.training.learning_rate
        )

    # scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.training.lr_reducing_factor,
        patience=params.training.patience
    )

    return Optimizer(optimizer, scheduler)
