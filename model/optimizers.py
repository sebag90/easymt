import torch


class Optimizer:
    def __init__(self, optimizer, scheduler, name):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.name = name

    def __repr__(self):
        return f"Optimizer({self.name} | lr: {round(self.lr, 5)})"

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def scheduler_step(self, loss):
        self.scheduler.step(loss)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, *args, **kwargs):
        self.optimizer.load_state_dict(*args, **kwargs)


class NoamOpt(Optimizer):
    """
    Optim wrapper that implements rate.
    """
    def __init__(self, model_size, factor, warmup, optimizer, step):
        super().__init__(optimizer, None, "Noam")
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()
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

        # calculate learning rate based on step
        d_size = self.model_size ** (-0.5)
        min_factor = min(step ** (-0.5), step * self.warmup ** (-1.5))

        return self.factor * (d_size * min_factor)


def get_optimizer(model, params):
    if params.training.optimizer == "noam":
        opt = NoamOpt(
            model_size=model.size,
            factor=params.training.noam_factor,
            warmup=params.training.warm_up,
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=0,
                betas=(0.9, 0.98),
                eps=1e-9),
            step=model.steps
            )

        return opt

    name = params.training.optimizer.capitalize()

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

    return Optimizer(optimizer, scheduler, name)
