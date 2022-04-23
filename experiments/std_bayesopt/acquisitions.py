import types
import torch

from helpers import ConformalSingleTaskGP, construct_conformal_bands


def conformalize_acq_fn(acq_fn_cls):
    old_forward = acq_fn_cls.forward

    def new_forward(self, X, *args, **kwargs):
        if not isinstance(self.model, ConformalSingleTaskGP):
            raise NotImplementedError(
                "Conformalized acquisitions can only be used with ConformalSingleTaskGP."
            )

        old_model = self.model
        target_grid, conf_pred_mask, conditioned_model, _, _ = construct_conformal_bands(
            old_model, X, old_model.alpha, old_model.temp
        )
        conditioned_model.standard()
        self.model = conditioned_model
        res = old_forward(X.unsqueeze(-3), *args, **kwargs)
        res = torch.trapezoid(
            y=conf_pred_mask * res[..., None, None], x=target_grid, dim=-3
        )
        self.model = old_model
        return res.view(-1)

    acq_fn_cls.forward = types.MethodType(new_forward, acq_fn_cls)
    return acq_fn_cls
