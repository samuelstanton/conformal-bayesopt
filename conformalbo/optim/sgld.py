import math
import torch
from torch import Tensor
from torch.optim import SGD
from typing import List


def sgld(params: List[Tensor],
         d_p_list: List[Tensor],
         momentum_buffer_list: List[Tensor],
         *,
         weight_decay: float,
         lr: float,
         momentum: float,
         noise: bool,
         temperature: float):
    r"""Functional API for SGMCMC/SGHMC.

    .. _SGLD\: Bayesian Learning via Stochastic Gradient Langevin Dynamics:
          https://icml.cc/2011/papers/398_icmlpaper.pdf
    .. _SGHMC\: Stochastic Gradient Hamiltonian Monte Carlo:
          http://www.istc-cc.cmu.edu/publications/papers/2014/Guestrin-stochastic-gradient.pdf
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
    
        if weight_decay != 0:
            d_p.add_(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            buf.mul_(1 - momentum).add_(d_p, alpha=-lr)
            if noise:
                eps = torch.randn_like(d_p)
                buf.add_(eps, alpha=math.sqrt(2 * lr * momentum * temperature))

            param.add_(buf)
        else:
            param.add_(d_p, alpha=-lr)

            if noise:
                eps = torch.randn_like(d_p)
                param.add_(eps, alpha=math.sqrt(2 * lr * temperature))


class SGLD(SGD):
    """Implements SGLD/SGHMC updates.

    Assumes negative log density.
    
    SGHMC updates are used for non-zero momentum values. The gradient noise
    variance is assumed to be zero. Mass matrix is kept to be identity.
    
    WARN: The variance estimate of gradients is assumed to be zero for SGHMC.
    """
    def __init__(self, *args, momentum=0, temperature=1, **kwargs):
        super().__init__(*args, momentum=momentum, **kwargs)

        self.T = temperature
        if momentum != 0:
            self.reset_momentum()

    @torch.no_grad()
    def step(self, closure=None, noise=True):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgld(params_with_grad,
                 d_p_list,
                 momentum_buffer_list,
                 weight_decay=weight_decay,
                 lr=lr,
                 momentum=momentum,
                 noise=noise,
                 temperature=self.T)

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    @torch.no_grad()
    def reset_momentum(self):
        for group in self.param_groups:
            momentum = group['momentum']

            assert momentum > 0, "Must use momentum > 0 to use SGHMC."

            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p)

        return self