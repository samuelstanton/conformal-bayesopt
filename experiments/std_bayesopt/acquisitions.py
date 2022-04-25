import types
import torch

from experiments.std_bayesopt.helpers import (
    # ConformalSingleTaskGP,
    construct_conformal_bands,
    conf_mask_to_bounds,
    sample_grid_points,
    conformal_gp_regression,
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

        # if not isinstance(self.model, ConformalSingleTaskGP):
        #     raise NotImplementedError(
        #         "Conformalized acquisitions can only be used with ConformalSingleTaskGP."
        #     )

        old_model = self.model
        q_batch_size = X.size(-2)

        if q_batch_size == 1:
            # compute conformal prediction mask on 1D dense grid
            target_grid, conf_pred_mask, conditioned_model = construct_conformal_bands(
                old_model, X, alpha, temp, grid_res, max_grid_refinements, ratio_estimator
            )
            # reshape X to match conditioned_model batch shape
            reshaped_x = X[..., None, None, :]
        else:
            # compute pointwise conformal prediction masks on dense grid
            with torch.no_grad():
                target_grid, conf_pred_mask, _ = construct_conformal_bands(
                    old_model, X, alpha, temp=1e-6, grid_res=grid_res,
                    max_grid_refinements=max_grid_refinements, ratio_estimator=ratio_estimator
                )
            # sample joint grid points independently from pointwise conformal intervals
            conf_lb, conf_ub = conf_mask_to_bounds(target_grid, conf_pred_mask)
            grid_samples = sample_grid_points(conf_lb, conf_ub, grid_res)
            # compute conformal prediction mask on joint grid
            conf_pred_mask, conditioned_model, _ = conformal_gp_regression(
                old_model, X, grid_samples, alpha, temp, ratio_estimator
            )
            # reshaping for consistency
            conf_pred_mask = conf_pred_mask.unsqueeze(-1)  # create target dim
            reshaped_x = X.unsqueeze(-3)
            reshaped_x = reshaped_x.expand(*[-1]*(X.ndim - 2), conf_pred_mask.size(-3), -1, -1)

        # temporarily overwrite model attribute
        # conditioned_model.standard()
        self.model = conditioned_model
        res = old_forward(reshaped_x, *args, **kwargs)

        # evaluate outer integral
        if q_batch_size == 1:
            res = torch.trapezoid(
                y=conf_pred_mask * res[..., None, None], x=target_grid, dim=-3
            ).view(-1)
        else:
            conf_pred_mask = conf_pred_mask.prod(dim=-2, keepdim=True)
            res = (conf_pred_mask * res[..., None, None]).mean(-3).view(-1)

        self.model = old_model

        return res

    acq_obj.forward = types.MethodType(new_forward, acq_obj)
    return acq_obj
