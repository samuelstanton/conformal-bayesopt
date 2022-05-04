import types
import torch

from torch import Tensor
from torch.nn import functional as F

from botorch import settings
from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
from botorch.acquisition.monte_carlo import (
    qUpperConfidenceBound,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    _split_fantasy_points,
    _get_value_function,
)
from botorch.utils.transforms import (
    match_batch_shape,
    t_batch_mode_transform,
    concatenate_pending_points,
)


from helpers import (
    construct_conformal_bands,
)


class ConformalAcquisition(object):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        self.alpha = alpha
        self.temp = temp
        self.grid_res = grid_res
        self.max_grid_refinements = max_grid_refinements
        self.ratio_estimator = ratio_estimator
        self.optimistic = optimistic
        self.grid_sampler = SobolQMCNormalSampler(grid_res) if grid_sampler is None else grid_sampler

    def _conformalize_model(self, X):
        if self.model is None:
            raise AttributeError("ConformalAcquisition must be subclassed")

        target_grid, grid_logp, conf_pred_mask, conditioned_model = construct_conformal_bands(
            self.model, X, self.alpha, self.temp, self.grid_res, self.max_grid_refinements,
            self.grid_sampler, self.ratio_estimator
        )
        return target_grid, grid_logp, conf_pred_mask, conditioned_model

    def _nonconformal_fwd(self, X, conditioned_model):
        raise NotImplementedError("ConformalAcquisition must be subclassed")

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
        values = self._nonconformal_fwd(reshaped_x, conditioned_model)
        # integrate w.r.t. batch outcome
        res = _conformal_integration(values, conf_pred_mask, grid_logp, self.alpha, opt_mask)

        return res.view(*orig_batch_shape)

    def forward(self, X):
        res = self._conformal_fwd(X)
        return res.max(dim=-1)[0]


class qConformalExpectedImprovement(ConformalAcquisition, qExpectedImprovement):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qExpectedImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _nonconformal_fwd(self, X, conditioned_model):
        posterior = conditioned_model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0.)
        q_ei = obj.mean(dim=0)
        return q_ei

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return ConformalAcquisition.forward(self, X)


class qConformalNoisyExpectedImprovement(ConformalAcquisition, qNoisyExpectedImprovement):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qNoisyExpectedImprovement.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _nonconformal_fwd(self, X, conditioned_model):
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = conditioned_model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        # note: don't use _forward_cached
        # if self._cache_root:
        #     diffs = self._forward_cached(posterior=posterior, X_full=X_full, q=q)
        # else:
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X_full)
        diffs = obj[..., -q:] - obj[..., :-q].max(dim=-1, keepdim=True).values

        return diffs.clamp_min(0).mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return ConformalAcquisition.forward(self, X)


class qConformalUpperConfidenceBound(ConformalAcquisition, qUpperConfidenceBound):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=True, grid_sampler=None, *args, **kwargs):
        qUpperConfidenceBound.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    def _nonconformal_fwd(self, X, conditioned_model):
        posterior = conditioned_model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = mean + self.beta_prime * (obj - mean).abs()
        return ucb_samples.mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X):
        return ConformalAcquisition.forward(self, X)


class qConformalKnowledgeGradient(ConformalAcquisition, qKnowledgeGradient):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
                 optimistic=False, grid_sampler=None, *args, **kwargs):
        qKnowledgeGradient.__init__(self, *args, **kwargs)
        ConformalAcquisition.__init__(
            self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator,
            optimistic, grid_sampler
        )

    @t_batch_mode_transform()
    def forward(self, X):
        q_batch_shape = X.shape[:-2]
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        target_grid, grid_logp, conf_pred_mask, conditioned_model = self._conformalize_model(X_actual)

        # get the value function
        value_function = _get_value_function(
            model=conditioned_model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            sampler=self.inner_sampler,
        )

        # reshape X_fantasies to match conditioned_model batch shape
        reshaped_x = X_fantasies.unsqueeze(-3)
        grid_res = conf_pred_mask.size(-3)
        reshaped_x = reshaped_x.expand(*[-1]*(X_fantasies.ndim - 2), grid_res, -1, -1)

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=reshaped_x)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        # average over the fantasy samples
        values = values.mean(0)
        values = values.view(*q_batch_shape, grid_res, -1)

        #
        conf_pred_mask = conf_pred_mask.prod(dim=-2, keepdim=True)
        opt_mask = torch.ones_like(conf_pred_mask)

        # integrate w.r.t. X_actual outcome variables
        res = _conformal_integration(values, conf_pred_mask, grid_logp, self.alpha, opt_mask)
        res = res.max(dim=-1)[0]
        res = res.view(*q_batch_shape)

        return res


def _conformal_integration(values, conf_pred_mask, grid_logp, alpha, opt_mask):
    """
    integrate w.r.t. outcome variables
    """
    combined_mask = conf_pred_mask * opt_mask

    with torch.no_grad():
        # count total number of optimistic outcomes
        opt_mask_weight = opt_mask.sum(-3, keepdim=True) + 1e-6
        nonconf_weights = opt_mask / opt_mask_weight
        # importance weights
        conf_weights = 1. / (grid_logp.exp() + 1e-6)
        # count number of optimistic outcomes in conformal set
        scaling_factor = (combined_mask * conf_weights).sum(dim=-3, keepdim=True)
        # normalize importance weights
        conf_weights = conf_weights / (scaling_factor + 1e-6)

    combined_weights = alpha * nonconf_weights + (1. - alpha) * combined_mask * conf_weights

    values = (combined_weights * values[..., None]).sum(-3)
    return values


# def conformalize_acq_fn(acq_obj, alpha, temp, grid_res, max_grid_refinements, ratio_estimator=None,
#                         optimistic=False, grid_sampler=None):
#     """
#     Returns conformalized BoTorch acquisition object
#     by monkey-patching the forward method.
#     Args:
#         acq_obj
#         alpha: float
#         temp: float
#         grid_res: int
#     Returns:
#         acq_fn_cls
#     """
#     old_forward = acq_obj.forward
#     if grid_sampler is None:
#         acq_obj.grid_sampler = SobolQMCNormalSampler(grid_res, resample=False)
#     else:
#         acq_obj.grid_sampler = grid_sampler
#
#     def new_forward(self, X, *args, **kwargs):
#         """
#         Args:
#             X: (*q_batch_shape, q_batch_size, input_dim)
#         Returns:
#             res: (*q_batch_shape,)
#         """
#         shape_msg = "inputs should have shape (*q_batch_shape, q_batch_size, input_dim)" \
#                     f"instead has shape {X.shape}"
#         assert X.ndim >= 3, shape_msg
#
#         old_model = self.model
#
#         # non-conformal forward pass first
#         # nonconf_res = old_forward(X, *args, **kwargs)
#
#         target_grid, grid_logp, conf_pred_mask, conditioned_model = construct_conformal_bands(
#             old_model, X, alpha, temp, grid_res, max_grid_refinements, self.grid_sampler, ratio_estimator
#         )
#
#         # reshape X to match conditioned_model batch shape
#         reshaped_x = X.unsqueeze(-3)
#         reshaped_x = reshaped_x.expand(*[-1]*(X.ndim - 2), conf_pred_mask.size(-3), -1, -1)
#
#         # temporarily overwrite model attribute
#         self.model = conditioned_model
#         res = old_forward(reshaped_x, *args, **kwargs)
#
#         # if optimistic only consider better than expected outcomes
#         if optimistic:
#             with torch.no_grad():
#                 tgt_post_mean = old_model.posterior(X).mean
#             opt_mask = (target_grid >= tgt_post_mean.unsqueeze(-3)).float()
#         else:
#             opt_mask = torch.ones_like(target_grid)
#
#         res = _conformal_integration(res, conf_pred_mask, grid_logp, alpha, opt_mask).view(-1)
#
#         self.model = old_model
#
#         return res
#
#     acq_obj.forward = types.MethodType(new_forward, acq_obj)
#     return acq_obj


# class qConformalUpperConfidenceBound(qUpperConfidenceBound):
#     def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.alpha = alpha
#         self.temp = temp
#         self.grid_res = grid_res
#         self.max_grid_refinements = max_grid_refinements
#         self.ratio_estimator = ratio_estimator
#         self.grid_sampler = SobolQMCNormalSampler(grid_res)
#
#     @concatenate_pending_points
#     @t_batch_mode_transform(assert_output_shape=False)
#     def forward(self, X: Tensor) -> Tensor:
#         r"""Evaluate qUpperConfidenceBound on the candidate set `X`.
#
#         Args:
#             X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
#                 points each.
#
#         Returns:
#             A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
#             design points `X`, where `batch_shape'` is the broadcasted batch shape of
#             model and input `X`.
#         """
#         posterior = self.model.posterior(
#             X=X, posterior_transform=self.posterior_transform
#         )
#         samples = self.sampler(posterior)
#         obj = self.objective(samples, X=X)
#         mean = obj.mean(dim=0)
#         ucb_samples = mean + self.beta_prime * (obj - mean).abs()
#         return ucb_samples.mean(dim=0)


# class qConformalKnowledgeGradient(qKnowledgeGradient):
#     def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.alpha = alpha
#         self.temp = temp
#         self.grid_res = grid_res
#         self.max_grid_refinements = max_grid_refinements
#         self.ratio_estimator = ratio_estimator
#         self.grid_sampler = SobolQMCNormalSampler(grid_res)
#
#     @t_batch_mode_transform()
#     def forward(self, X: Tensor) -> Tensor:
#         r"""Evaluate qKnowledgeGradient on the candidate set `X`.
#
#         Args:
#             X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
#                 `q + num_fantasies` design points each. We split this X tensor
#                 into two parts in the `q` dimension (`dim=-2`). The first `q`
#                 are the q-batch of design points and the last num_fantasies are
#                 the current solutions of the inner optimization problem.
#
#                 `X_fantasies = X[..., -num_fantasies:, :]`
#                 `X_fantasies.shape = b x num_fantasies x d`
#
#                 `X_actual = X[..., :-num_fantasies, :]`
#                 `X_actual.shape = b x q x d`
#
#         Returns:
#             A Tensor of shape `b`. For t-batch b, the q-KG value of the design
#                 `X_actual[b]` is averaged across the fantasy models, where
#                 `X_fantasies[b, i]` is chosen as the final selection for the
#                 `i`-th fantasy model.
#                 NOTE: If `current_value` is not provided, then this is not the
#                 true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
#                 maximized at fixed `X_actual[b]`.
#         """
#         X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)
#
#         # We only concatenate X_pending into the X part after splitting
#         if self.X_pending is not None:
#             X_actual = torch.cat(
#                 [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
#             )
#
#         target_grid, grid_logp, conf_pred_mask, fantasy_model = construct_conformal_bands(
#             self.model,
#             X_actual,
#             self.alpha,
#             self.temp,
#             self.grid_res,
#             self.max_grid_refinements,
#             self.grid_sampler,
#             self.ratio_estimator,
#         )
#
#         # get the value function
#         value_function = _get_value_function(
#             model=fantasy_model,
#             objective=self.objective,
#             posterior_transform=self.posterior_transform,
#             sampler=self.inner_sampler,
#         )
#
#         # reshape X_fantasies to match conditioned_model batch shape
#         reshaped_x = X_fantasies.unsqueeze(-3)
#         reshaped_x = reshaped_x.expand(*[-1]*(X_fantasies.ndim - 2), conf_pred_mask.size(-3), -1, -1)
#
#         # make sure to propagate gradients to the fantasy model train inputs
#         with settings.propagate_grads(True):
#             values = value_function(X=reshaped_x)  # num_fantasies x b
#
#         if self.current_value is not None:
#             values = values - self.current_value
#
#         # average over the fantasy samples
#         values = values.mean(0)
#         res = _conformal_integration(values, conf_pred_mask, grid_logp, self.alpha).view(-1)
#
#         # point_weights = 1. / (grid_logp.exp().detach() + 1e-6)
#         # # point_weights =
#         # conf_pred_mask = conf_pred_mask.prod(dim=-2, keepdim=True)
#         # mask_weight = conf_pred_mask.sum(-3).detach() + 1e-6
#         # res = (conf_pred_mask * point_weights * values[..., None, None]).sum(-3)
#         # res = (res / mask_weight).view(-1)
#
#         return res
