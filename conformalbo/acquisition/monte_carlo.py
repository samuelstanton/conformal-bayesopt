import torch

from torch import Tensor

from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)


class qSmoothExpectedImprovement(qExpectedImprovement):
    def __init__(self, temp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        obj = obj - self.best_f.unsqueeze(-1).to(obj)
        obj = torch.stack([obj, torch.zeros_like(obj)], dim=-1)
        # smooth approximation to max operators
        obj = softplus(obj, -1, self.temp)
        q_ei = softplus(obj, -1, self.temp).mean(0)

        return q_ei


class qSmoothNoisyExpectedImprovement(qNoisyExpectedImprovement):
    def __init__(self, temp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if self._cache_root:
            diffs = self._forward_cached(posterior=posterior, X_full=X_full, q=q)
        else:
            samples = self.sampler(posterior)
            obj = self.objective(samples, X=X_full)
            diffs = (
                softplus(obj[..., -q:], -1, self.temp) - softplus(obj[..., :-q], -1, self.temp)
            )
        diffs = torch.stack([diffs, torch.zeros_like(diffs)], -1)
        q_nei = softplus(diffs, -1, self.temp).mean(0)

        return q_nei


class qSmoothUpperConfidenceBound(qUpperConfidenceBound):
    def __init__(self, temp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = temp

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = mean + self.beta_prime * (obj - mean).abs()
        qucb = softplus(ucb_samples, -1, self.temp).mean(0)
        return qucb


def softplus(tensor, dim, temp, keepdim=False):
    return temp * torch.logsumexp(tensor / temp, dim, keepdim=keepdim)
