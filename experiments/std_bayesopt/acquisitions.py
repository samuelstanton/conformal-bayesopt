import types
import torch

from torch import Tensor

from botorch import settings
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    _split_fantasy_points,
    _get_value_function,
)
from botorch.utils.transforms import (
    match_batch_shape,
    t_batch_mode_transform,
)


from experiments.std_bayesopt.helpers import (
    construct_conformal_bands,
)


def conformalize_acq_fn(acq_obj, alpha, temp, grid_res, max_grid_refinements, ratio_estimator=None):
    """
    Returns conformalized BoTorch acquisition object
    by monkey-patching the forward method.
    Args:
        acq_obj
        alpha: float
        temp: float
        grid_res: int
    Returns:
        acq_fn_cls
    """
    old_forward = acq_obj.forward
    acq_obj.grid_sampler = SobolQMCNormalSampler(grid_res, resample=False)

    def new_forward(self, X, *args, **kwargs):
        """
        Args:
            X: (*q_batch_shape, q_batch_size, input_dim)
        Returns:
            res: (*q_batch_shape,)
        """
        shape_msg = "inputs should have shape (*q_batch_shape, q_batch_size, input_dim)" \
                    f"instead has shape {X.shape}"
        assert X.ndim >= 3, shape_msg

        old_model = self.model

        target_grid, conf_pred_mask, conditioned_model = construct_conformal_bands(
            old_model, X, alpha, temp, grid_res, max_grid_refinements, self.grid_sampler, ratio_estimator
        )
        # reshape X to match conditioned_model batch shape
        reshaped_x = X.unsqueeze(-3)
        reshaped_x = reshaped_x.expand(*[-1]*(X.ndim - 2), conf_pred_mask.size(-3), -1, -1)

        # temporarily overwrite model attribute
        self.model = conditioned_model
        res = old_forward(reshaped_x, *args, **kwargs)

        # evaluate outer integral
        # TODO remove when batch conformal scores are implemented
        conf_pred_mask = conf_pred_mask.prod(dim=-2, keepdim=True)
        # added epsilon due to removal of conf_pred_mask summation check
        weights = conf_pred_mask / (conf_pred_mask.sum(-3, keepdim=True) + 1e-6).detach()
        res = (weights * res[..., None, None]).sum(-3)

        self.model = old_model

        return res.view(-1)

    acq_obj.forward = types.MethodType(new_forward, acq_obj)
    return acq_obj


class qConformalKnowledgeGradient(qKnowledgeGradient):
    def __init__(self, alpha, temp, grid_res, max_grid_refinements, ratio_estimator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temp = temp
        self.grid_res = grid_res
        self.max_grid_refinements = max_grid_refinements
        self.ratio_estimator = ratio_estimator
        self.grid_sampler = SobolQMCNormalSampler(grid_res)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qKnowledgeGradient on the candidate set `X`.

        Args:
            X: A `b x (q + num_fantasies) x d` Tensor with `b` t-batches of
                `q + num_fantasies` design points each. We split this X tensor
                into two parts in the `q` dimension (`dim=-2`). The first `q`
                are the q-batch of design points and the last num_fantasies are
                the current solutions of the inner optimization problem.

                `X_fantasies = X[..., -num_fantasies:, :]`
                `X_fantasies.shape = b x num_fantasies x d`

                `X_actual = X[..., :-num_fantasies, :]`
                `X_actual.shape = b x q x d`

        Returns:
            A Tensor of shape `b`. For t-batch b, the q-KG value of the design
                `X_actual[b]` is averaged across the fantasy models, where
                `X_fantasies[b, i]` is chosen as the final selection for the
                `i`-th fantasy model.
                NOTE: If `current_value` is not provided, then this is not the
                true KG value of `X_actual[b]`, and `X_fantasies[b, : ]` must be
                maximized at fixed `X_actual[b]`.
        """
        X_actual, X_fantasies = _split_fantasy_points(X=X, n_f=self.num_fantasies)

        # We only concatenate X_pending into the X part after splitting
        if self.X_pending is not None:
            X_actual = torch.cat(
                [X_actual, match_batch_shape(self.X_pending, X_actual)], dim=-2
            )

        target_grid, conf_pred_mask, fantasy_model = construct_conformal_bands(
            self.model,
            X_actual,
            self.alpha,
            self.temp,
            self.grid_res,
            self.max_grid_refinements,
            self.grid_sampler,
            self.ratio_estimator,
        )

        # get the value function
        value_function = _get_value_function(
            model=fantasy_model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            sampler=self.inner_sampler,
        )

        # reshape X_fantasies to match conditioned_model batch shape
        reshaped_x = X_fantasies.unsqueeze(-3)
        reshaped_x = reshaped_x.expand(*[-1]*(X_fantasies.ndim - 2), conf_pred_mask.size(-3), -1, -1)

        # make sure to propagate gradients to the fantasy model train inputs
        with settings.propagate_grads(True):
            values = value_function(X=reshaped_x)  # num_fantasies x b

        if self.current_value is not None:
            values = values - self.current_value

        conf_pred_mask = conf_pred_mask.prod(dim=-2, keepdim=True)
        # added epsilon due to removal of conf_pred_mask summation check
        weights = conf_pred_mask / (conf_pred_mask.sum(-3, keepdim=True) + 1e-6).detach()
        res = (weights * values[..., None, None]).sum(-3)
        # return average over the fantasy samples
        res = res.mean(0).view(-1)

        return res
