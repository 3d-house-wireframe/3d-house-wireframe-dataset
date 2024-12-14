import torch.optim.lr_scheduler as lrs
import math
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.optim import Optimizer

class LinearHalfConsine(lrs.LambdaLR):
    def __init__(
        self, 
        optimizer = None, 
        warmup_epoch = 5, 
        max_epoch = 200, 
        min_lr = 1e-6, 
        lr=5e-5
    ):
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        self.early_epochs = warmup_epoch
        self.later_epochs = max_epoch - warmup_epoch
        self.min_coef = min_lr / lr
        
        def rule(epoch):
            if epoch < self.warmup_epoch:
                coef = epoch / self.early_epochs * (1 - self.min_coef) + self.min_coef
            else:
                coef = self.min_coef + \
                    0.5 * (1 - self.min_coef) * \
                        (1 + math.cos(math.pi * (epoch - self.warmup_epoch) / self.later_epochs))

            return coef

        super().__init__(optimizer, lr_lambda=rule)

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
