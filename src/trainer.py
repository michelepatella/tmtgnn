"""src/trainer.py

Training for T-MTGNN models.

This module provides the `Trainer` class, which encapsulates the logic
for a single training step, including forward propagation, loss computation,
backpropagation, gradient clipping, and parameter updates.
"""
import torch
from torch.nn.utils import clip_grad_norm_
from typing import Callable


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_grad_norm: float,
        device: str,
    ) -> None:
        """Trainer class for training T-MTGNN models.

        This class handles training of T-MTGNN models, including forward pass,
        loss computation, backpropagation, and gradient clipping.

        Args:
            model (torch.nn.Module):
                T-MTGNN model to train.
            optimizer (torch.optim.Optimizer):
                Optimizer used to update model parameters.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
                Loss function used during training which takes model outputs and
                targets as input and returns a scalar loss tensor (0-dimensional).
            max_grad_norm (float):
                Maximum gradient norm for gradient clipping.
            device (str):
                Device used for computation (e.g., ``"cpu"``, ``"cuda"``).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self.device = device

    def train(self, x: torch.Tensor, target: torch.Tensor) -> float:
        """Run a single training step.

        This method performs a forward pass, computes the loss, backpropagates
        gradients, applies gradient clipping, and updates the model parameters.

        Args:
            x (torch.Tensor): 
                Input tensor for the model.
            target (torch.Tensor): 
                Ground-truth tensor used to compute the loss.

        Returns:
            float: 
                Scalar loss value for the training step.
        """
        # Model and optimizer setup for training
        self.model.train()
        self.optimizer.zero_grad()

        # Data transfer to device
        x = x.to(self.device)
        target = target.to(self.device)

        # Forward pass
        output = self.model(x)

        # Loss computation
        loss = self.loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Parameters update
        self.optimizer.step()

        # Final loss value
        return loss.item()
