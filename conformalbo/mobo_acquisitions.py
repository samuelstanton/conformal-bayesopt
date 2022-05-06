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
from botorch.acquisition.multi_objective import (
   qExpectedHypervolumeImprovement,
   qNoisyExpectedHypervolumeImprovement,
)

from helpers import construct_conformal_bands

from acquisitions import ConformalAcquisition, _conformal_integration

class qConformalExpectedHypervolumeImprovement(ConformalAcquisition, qExpectedHypervolumeImprovement):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qExpectedHypervolumeImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )
        
        assert self.constraints is None # for simplicity of code

    def _nonconformal_fwd(self, X, conditioned_model):
          # we use _conformal_ehvi instead
        pass
    
    def _conformal_ehvi(self, samples, X, conformal_kwargs):
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.
        
        Mocks _compute_ehvi in qExpectedHypervolumeImprovement
        
        Args:
            samples: A `n_samples x batch_shape x q' x m`-dim tensor of samples.
            X: A `batch_shape x q x d`-dim tensor of inputs.
        Returns:
            A `batch_shape x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # unsqueeze these to enable expansion over the num_cells dimension
        conformal_kwargs["conf_pred_mask"] = conformal_kwargs["conf_pred_mask"].unsqueeze(0)
        conformal_kwargs["opt_mask"] = conformal_kwargs["opt_mask"].unsqueeze(0)
        conformal_kwargs["grid_logp"] = conformal_kwargs["grid_logp"].unsqueeze(0)
            
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]

        self._cache_q_subset_indices(q_out=q)
        # batch_shape = obj.shape[1:-3]
        batch_shape = obj.shape[:-2]
        # this is input_batch_shape x cell x 1
        # the 1 is here for the conformal shapes
#         areas_per_segment = torch.zeros(
#             *batch_shape,
#             self.cell_lower_bounds.shape[-2],
#             1,
#             dtype=obj.dtype,
#             device=obj.device,
#         )
        areas_per_segment = 0.
    
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        # the loop is the q batch integral
        for i in range(1, self.q_out + 1):
            q_choose_i = self.q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # substract cell lower bounds, clamp min at zero
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i
            areas_i = lengths_i.prod(dim=-1)
            
            # first reduce over the MC dim
            # this tensor is batch_shape x num_cells x q_choose_i
            # b/c this is inside the integral
            areas_i = (-1) ** (i + 1) * areas_i
            avged_areas_i = areas_i.mean(dim=0)
            
            # move the num_cells dim first to have sane integration
            avged_areas_i = avged_areas_i.movedim(-2, 0)
            
            # this tensor is num_cells x batch_shape x conformal grid x 1
            conformal_area_of_segment = _conformal_integration(
                avged_areas_i, **conformal_kwargs,
            )
            # and we sum over the q batch???
            conformal_area_of_segment = conformal_area_of_segment.sum(dim=-2)
            
            # now put the num_cells back
            # batch_shape x num_cells x 1
            conformal_area_of_segment = conformal_area_of_segment.movedim(0, -2)
            
            areas_per_segment = areas_per_segment + conformal_area_of_segment
            
        # sum over segments
        # areas integral and MC samples integral
        # we sum over -2 here bc the last dim is an output dim
        # this last sum reduces over the num_cells dimension
        return areas_per_segment.sum(dim=-2)
    
    def _conformal_fwd(self, X):
        orig_batch_shape = X.shape[:-1]
        target_grid, grid_logp, conf_pred_mask, conditioned_model = self._conformalize_model(X)

        if self.optimistic:
            with torch.no_grad():
                tgt_post_mean = self.model.posterior(X).mean
            opt_mask = (target_grid >= tgt_post_mean.unsqueeze(-3)).float()
        else:
            opt_mask = torch.ones_like(target_grid)

        # reshape X to match conditioned_model batch shape
        reshaped_x = X.unsqueeze(-3)
        reshaped_x = reshaped_x.expand(*[-1]*(X.ndim - 2), conf_pred_mask.size(-3), -1, -1)
        
        # standard forward pass using conditioned models
        # here the integration is nested inside of the conformal loop
        posterior = conditioned_model.posterior(reshaped_x)
        samples = self.sampler(posterior)
        
        conformal_integration_kwargs = {
            "conf_pred_mask": conf_pred_mask, 
            "grid_logp": grid_logp, 
            "alpha": self.alpha,
            "opt_mask": opt_mask,
        }
        res = self._conformal_ehvi(samples, reshaped_x, conformal_integration_kwargs)
        
        return res
    

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        # return ConformalAcquisition.forward(self, X)
        return self._conformal_fwd(X).squeeze(-1)


class qConformalNoisyExpectedHypervolumeImprovement(
    qConformalExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement
):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qNoisyExpectedHypervolumeImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _conformal_fwd(self, X):
        orig_batch_shape = X.shape[:-1]
        target_grid, grid_logp, conf_pred_mask, conditioned_model = self._conformalize_model(X)

        if self.optimistic:
            with torch.no_grad():
                tgt_post_mean = self.model.posterior(X).mean
            opt_mask = (target_grid >= tgt_post_mean.unsqueeze(-3)).float()
        else:
            opt_mask = torch.ones_like(target_grid)

        # reshape X to match conditioned_model batch shape
        reshaped_x = X.unsqueeze(-3)
        reshaped_x = reshaped_x.expand(*[-1]*(X.ndim - 2), conf_pred_mask.size(-3), -1, -1)
        
        X_full = torch.cat([match_batch_shape(self.X_baseline, reshaped_x), reshaped_x], dim=-2)
        # Note: it is important to compute the full posterior over `(X_baseline, X)`
        # to ensure that we properly sample `f(X)` from the joint distribution
        
        posterior = conditioned_model.posterior(X_full)
        # hardcode support for non fully bayesian models for the time being
        event_shape_lag = 2
        n_w = posterior.event_shape[X_full.dim() - event_shape_lag] // X_full.shape[-2]
        q_in = X.shape[-2] * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        
        conformal_integration_kwargs = {
            "conf_pred_mask": conf_pred_mask, 
            "grid_logp": grid_logp, 
            "alpha": self.alpha,
            "opt_mask": opt_mask,
        }
        res = self._conformal_ehvi(samples, reshaped_x, conformal_integration_kwargs)
        
        # Add previous nehvi from pending points.
        return res + self._prev_nehvi

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return self._conformal_fwd(X).squeeze(-1)
