"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register

__all__ = [
    "AdamW",
    "SGD",
    "Adam",
    "MultiStepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "LambdaLR",
    "LinearLR",
]


SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)

# PyTorch >= 1.13 provides a built-in "LinearLR" scheduler that linearly decays
# the learning-rate from `start_factor * lr` to `end_factor * lr` over
# `total_iters` calls to ``scheduler.step()``.  Add it to the registry so that
# it can be referenced from YAML configuration files the same way other
# schedulers are.

if hasattr(lr_scheduler, "LinearLR"):
    # Wrap the built-in implementation so that unknown keyword arguments are
    # ignored gracefully.  This makes the scheduler resilient to superfluous
    # fields that might leak in from included configuration files (e.g.
    # `milestones` or `gamma` coming from a previously used `MultiStepLR`).

    from inspect import signature

    _signature = signature(lr_scheduler.LinearLR)

    from typing import Any

    class _LinearLRBuiltin(lr_scheduler.LinearLR):  # type: ignore[misc]
        """Subclass of the built-in ``LinearLR`` that safely ignores unknown
        keyword arguments originating from YAML merges (e.g. ``milestones``).
        """

        def __init__(self, optimizer, *args: Any, **kwargs: Any):  # noqa: D401
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in _signature.parameters
            }
            super().__init__(optimizer, *args, **filtered_kwargs)

    LinearLR = register(name="LinearLR")(_LinearLRBuiltin)
else:
    # Fall back to a minimal re-implementation for older PyTorch versions. This
    # ensures the config does not break even if the runtime environment lacks
    # the upstream implementation.

    from torch.optim import Optimizer

    class _LinearLR(lr_scheduler._LRScheduler):  # noqa: D401, N801 – keep naming consistent with torch
        """Linear learning-rate scheduler.

        Decays the learning-rate linearly from ``start_factor`` to
        ``end_factor`` over ``total_iters`` steps and then keeps it constant.
        The implementation mirrors the behaviour of
        ``torch.optim.lr_scheduler.LinearLR`` introduced in PyTorch 1.13 so
        that we can rely on the same set of constructor arguments regardless
        of the library version present at runtime.
        """

        def __init__(
            self,
            optimizer: Optimizer,
            start_factor: float = 1.0,
            end_factor: float = 0.0,
            total_iters: int = 5,
            last_epoch: int = -1,
            **kwargs,
        ) -> None:  # noqa: D401
            if total_iters <= 0:
                raise ValueError("total_iters must be positive")
            self.start_factor = float(start_factor)
            self.end_factor = float(end_factor)
            self.total_iters = total_iters
            super().__init__(optimizer, last_epoch)

        def get_lr(self):  # type: ignore[override]
            if self.last_epoch >= self.total_iters:
                factor = self.end_factor
            else:
                factor = self.start_factor + (
                    (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
                )
            return [base_lr * factor for base_lr in self.base_lrs]

    LinearLR = register(name="LinearLR")(_LinearLR)
