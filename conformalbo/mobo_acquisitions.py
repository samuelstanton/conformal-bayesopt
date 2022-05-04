import types
import torch

from torch import Tensor
from torch.nn import functional as F

from botorch import settings
from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils.transforms import (
    match_batch_shape,
    t_batch_mode_transform,
    concatenate_pending_points,
)


from helpers import (
    construct_conformal_bands,
)


from botorch.acquisition.multi_objective import (
   qExpectedHypervolumeImprovement,
   qNoisyExpectedHypervolumeImprovement,
)

from acquisitions import ConformalAcquisition

class qConformalExpectedHypervolumeImprovement(ConformalAcquisition, qExpectedHypervolumeImprovement):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qExpectedHypervolumeImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _nonconformal_fwd(self, X, conditioned_model):
        # old_model = self.model
        # self.model = conditioned_model
        # TODO: can we just make this work
        # res = qExpectedHypervolumeImprovement.forward(self, X)
        posterior = conditioned_model.posterior(X)
        samples = self.sampler(posterior).unsqueeze(-2)
        res = self._compute_qehvi(samples=samples, X=X.unsqueeze(-2))
    
        # self.model = old_model
        return res

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return ConformalAcquisition.forward(self, X)


class qConformalNoisyExpectedHypervolumeImprovement(ConformalAcquisition, qNoisyExpectedHypervolumeImprovement):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qNoisyExpectedHypervolumeImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _nonconformal_fwd(self, X, conditioned_model):
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # Note: it is important to compute the full posterior over `(X_baseline, X)`
        # to ensure that we properly sample `f(X)` from the joint distribution `
        # `f(X_baseline, X) ~ P(f | D)` given that we can already fixed the sampled
        # function values for `f(X_baseline)`.
        # TODO: improve efficiency by not recomputing baseline-baseline
        # covariance matrix.
        posterior = conditioned_model.posterior(X_full)
        # Account for possible one-to-many transform and the MCMC batch dimension in
        # `SaasFullyBayesianSingleTaskGP`
        # hardcode support for non fully bayesian models for the time being
        # event_shape_lag = 1 if is_fully_bayesian(self.model) else 2
        event_shape_lag = 2
        n_w = posterior.event_shape[X_full.dim() - event_shape_lag] // X_full.shape[-2]
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        
        X = X.unsqueeze(-2)
        samples = samples.unsqueeze(-2)
        
        # Add previous nehvi from pending points.
        return self._compute_qehvi(samples=samples, X=X) + self._prev_nehvi
    
#     def _nonconformal_fwd(self, X, conditioned_model):
#         old_model = self.model
#         self.model = conditioned_model
#         res = qNoisyExpectedHypervolumeImprovement.forward(self, X)
#         self.model = old_model
#         return res

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return ConformalAcquisition.forward(self, X)
