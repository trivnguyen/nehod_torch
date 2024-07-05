
import torch
import torch.nn as nn
import math

def get_activation(activation):
    """ Get an activation function. """
    if activation.name.lower() == 'identity':
        return nn.Identity()
    elif activation.name.lower() == 'relu':
        return nn.ReLU(inplace=False)
    elif activation.name.lower() == 'tanh':
        return nn.Tanh()
    elif activation.name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.name.lower() == 'leaky_relu':
        return nn.LeakyReLU(activation.leaky_relu_alpha, inplace=False)
    elif activation.name.lower() == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation function: {activation.name}')

class WarmUpCosineDecayLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self, optimizer, init_value, peak_value, warmup_steps, decay_steps,
        end_value=0.0, exponent=1.0, last_epoch=-1):
        self.init_value = init_value
        self.peak_value = peak_value
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_value = end_value
        self.exponent = exponent

        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def cosine_decay_schedule(self, step, init_value, decay_steps, alpha=0.0, exponent=1.0):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decayed = (1 - alpha) * cosine_decay ** exponent + alpha
        return init_value * decayed

    def linear_schedule(self, step, init_value, peak_value, warmup_steps):
        return init_value + (peak_value - init_value) * step / warmup_steps

    def lr_lambda(self, step):
        alpha = 0 if self.peak_value == 0 else self.end_value / self.peak_value
        if step < self.warmup_steps:
            return self.linear_schedule(
                step, self.init_value, self.peak_value, self.warmup_steps)
        return self.cosine_decay_schedule(
            step - self.warmup_steps,
            init_value=self.peak_value,
            decay_steps=self.decay_steps - self.warmup_steps,
            alpha=alpha,
            exponent=self.exponent,
        )

def configure_optimizers(parameters, optimizer_args, scheduler_args):
    """ Return optimizer and scheduler. """

    # setup the optimizer
    if optimizer_args.name == "Adam":
        return torch.optim.Adam(
            parameters(), lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay)
    elif optimizer_args.name == "AdamW":
        return torch.optim.AdamW(
            parameters(), lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay)
    else:
        raise NotImplementedError(
            "Optimizer {} not implemented".format(optimizer_args.name))

    # setup the scheduler
    if scheduler_args.get(name) is None:
        scheduler = None
    elif scheduler_args.name == 'ReduceLROnPlateau':
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=scheduler_args.factor,
            patience=scheduler_args.patience)
    elif scheduler_args.name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_args.T_max,
            eta_min=scheduler_args.eta_min)
    elif scheduler_args.name == 'WarmUpCosineDecayLR':
        scheduler = WarmUpCosineDecayLR(
            optimizer,
            init_value=scheduler_args.init_value,
            peak_value=scheduler_args.peak_value,
            warmup_steps=scheduler_args.warmup_steps,
            decay_steps=scheduler_args.decay_steps,
        )
    else:
        raise NotImplementedError(
            "Scheduler {} not implemented".format(scheduler_args.name))

    if scheduler is None:
        return optimizer
    else:
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': scheduler_args.interval,
                'frequency': 1
            }
        }
