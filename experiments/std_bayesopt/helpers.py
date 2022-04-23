import numpy as np
import torch

from botorch.posteriors import Posterior
from botorch.sampling import MCSampler
from botorch.models import SingleTaskGP

from scipy import stats

import torchsort


def generate_target_grid(y_mean, y_std, grid_res, alpha):
    """
    Args:
        y_mean: (*batch_shape, q_batch_size, y_dim)
        y_std: (*batch_shape, q_batch_size, y_dim)
        grid_res: float
        alpha: float
    Returns:
        target_grid: (*batch_shape, grid_res, q_batch_size, y_dim)
    """
    cred_levels = np.linspace(alpha / 2.0, 1 - alpha / 2.0, grid_res)
    std_factors = stats.norm.ppf(cred_levels)[..., None, None]  # (grid_res, 1, 1)
    std_factors = torch.from_numpy(std_factors).to(y_mean)
    scaled_std = std_factors * y_std
    if scaled_std.ndim > y_mean.ndim:
        y_mean = y_mean.unsqueeze(-3)
    y_grid = y_mean + scaled_std
    return y_grid


def construct_conformal_bands(model, inputs, alpha, temp=1e-6):
    # generate conformal prediction mask
    if inputs.ndim == 2:
        inputs = inputs[:, None]

    model.eval()
    model.standard()
    with torch.no_grad():
        y_post = model.posterior(inputs, observation_noise=True)
        y_mean = y_post.mean[..., None]
        y_std = y_post.variance.sqrt()[..., None]

    grid_res = model.tgt_grid_res
    grid_center = y_mean
    grid_scale = y_std
    assert grid_res >= 2
    cred_tail_prob = alpha

    for refine_step in range(1, model.max_grid_refinements + 1):
        target_grid = generate_target_grid(
            grid_center, grid_scale, grid_res, alpha=cred_tail_prob
        )
        conf_pred_mask, conditioned_gps, q_conf_scores = conformal_gp_regression(
            model, inputs, target_grid, alpha, temp=temp
        )

        accepted_ratios = (conf_pred_mask > 0.5).float().mean(1)

        # center target grid around best scoring point in current grid
        with torch.no_grad():
            q_score_argmax = q_conf_scores.argmax(
                dim=-2, keepdim=True
            )  # (*q_batch_shape, 1, q_batch_size)
            new_center = torch.gather(
                target_grid, dim=-3, index=q_score_argmax[..., None]
            )
            new_center = new_center.squeeze(
                -3
            )  # (*q_batch_shape, q_batch_size, target_dim)
        if not torch.allclose(grid_center, new_center, rtol=1e-2):
            recenter_grid = True
            grid_center = new_center
        else:
            recenter_grid = False

        # check if grid resolution should be increased
        too_few_accepted = accepted_ratios < 0.2
        if torch.any(too_few_accepted):
            refine_grid = True
            grid_res *= 2
            # print('increasing grid resolution')
        else:
            refine_grid = False

        # if too many/all grid elements are in prediction set, expand the grid
        too_many_accepted = accepted_ratios > 0.8
        if (
            torch.any(conf_pred_mask[:, 0] > 0.5)
            or torch.any(conf_pred_mask[:, -1] > 0.5)
            or torch.any(too_many_accepted)
        ):
            expand_grid = True
            grid_scale *= 2
        else:
            expand_grid = False

        if not any([recenter_grid, refine_grid, expand_grid]):
            break

        if refine_step < model.max_grid_refinements:
            del conditioned_gps

    # print(
    #     f"conformal step: {refine_step},",
    #     f"grid resolution: {grid_res},",
    #     f"tail_prob: {cred_tail_prob:0.4f},",
    #     f"accepted ratio: {accepted_ratios.mean().item():0.2f}"
    # )

    # TODO improve error handling for case when max_iter is exhausted
    if torch.any((conf_pred_mask > 0.5).float().sum(1) < 2):
        raise RuntimeError("not enough grid points accepted")

    # convert conformal prediction mask to upper and lower bounds
    conf_pred_mask = conf_pred_mask.view(
        *target_grid.shape
    )  # (num_inputs, num_grid_pts, tgt_dim)

    conf_ub = (
        torch.stack(
            [
                grid[mask > 0.5].max(0)[0]
                for grid, mask in zip(target_grid, conf_pred_mask)
            ]
        )
        .cpu()
        .view(-1)
    )
    conf_lb = (
        torch.stack(
            [
                grid[mask > 0.5].min(0)[0]
                for grid, mask in zip(target_grid, conf_pred_mask)
            ]
        )
        .cpu()
        .view(-1)
    )

    return target_grid, conf_pred_mask, conditioned_gps, conf_lb, conf_ub


def conformal_gp_regression(
    gp, test_inputs, target_grid, alpha, temp=1e-6, ratio_estimator=None, **kwargs
):
    """
    Full conformal Bayes for exact GP regression.
    Args:
        gp (gpytorch.models.GP)
        test_inputs (torch.Tensor): (*q_batch_shape, q_batch_size, input_dim)
        target_grid (torch.Tensor): (*q_batch_shape, num_grid_pts, q_batch_size, target_dim)
        alpha (float)
        ratio_estimator (torch.nn.Module)
    Returns:
        conf_pred_mask (torch.Tensor): (*q_batch_shape, num_grid_pts, q_batch_size)
    """

    shape_msg = "test inputs should have shape (*q_batch_shape, q_batch_size, input_dim)"
    assert test_inputs.ndim == 3, shape_msg
    q_batch_shape = test_inputs.shape[:-2]
    grid_size, q_batch_size, target_dim = target_grid.shape[-3:]

    gp.eval()
    gp.conf_pred_mask = (
        None  # without this line the deepcopy in `gp.condition_on_observations` fails
    )
    # code smell reminder:
    gp.conformal()  # fix for BoTorch acq fn batch shape checks

    # retraining: condition the GP at every target grid point for every test input
    expanded_inputs = test_inputs.unsqueeze(-3).expand(
        *[-1] * len(q_batch_shape), target_grid.shape[1], -1, -1
    )  # (*q_batch_shape, num_grid_pts, q_batch_size, input_dim)

    updated_gps = gp.condition_on_observations(expanded_inputs, target_grid)

    # get ready to compute the conformal scores
    num_old_train = gp.train_inputs[0].size(-2)
    num_total = updated_gps.train_inputs[0].size(-2)  # num_old_train + q_batch_size
    assert num_total == num_old_train + q_batch_size

    train_inputs = updated_gps.train_inputs[
        0
    ]  # (*q_batch_shape, grid_size, num_total, input_dim)
    train_labels = updated_gps.prediction_strategy.train_labels
    train_labels = train_labels.view(*q_batch_shape, grid_size, num_total, target_dim)

    if hasattr(updated_gps, "input_transform"):
        train_inputs = updated_gps.input_transform.untransform(train_inputs)
    if hasattr(updated_gps, "output_transform"):
        train_labels = updated_gps.output_transform.untransform(train_labels)

    # compute conformal scores (pointwise posterior predictive log-likelihood)
    # TODO extend to target_dim > 1 case
    if target_dim > 1:
        raise NotImplementedError
    updated_gps.standard()
    posterior = updated_gps.posterior(train_inputs, observation_noise=True)
    pred_dist = torch.distributions.Normal(posterior.mean, posterior.variance.sqrt())
    conf_scores = pred_dist.log_prob(train_labels)
    conf_scores = conf_scores.view(*q_batch_shape, grid_size, num_total)
    q_conf_scores = conf_scores[..., num_old_train:].detach()

    original_shape = conf_scores.shape
    ranks_by_score = torchsort.soft_rank(
        conf_scores.flatten(0, -2),
        regularization="l2",
        regularization_strength=0.1,
    ).view(
        *original_shape
    )  # (num_q_batches, grid_size, num_total)
    threshold = ranks_by_score[..., num_old_train:]
    rank_mask = 1 - torch.sigmoid(
        (ranks_by_score.unsqueeze(-1) - threshold.unsqueeze(-2)) / temp
    )  # (num_q_batches, grid_size, num_total, q_batch_size)
    rank_mask[..., np.arange(-q_batch_size, 0), np.arange(-q_batch_size, 0)] *= 2

    if ratio_estimator is None:
        imp_weights = torch.zeros_like(rank_mask, requires_grad=False)
        imp_weights[..., :num_old_train, :] = 1.0 / (num_old_train + 1)
        imp_weights[
            ..., np.arange(-q_batch_size, 0), np.arange(-q_batch_size, 0)
        ] = 1.0 / (num_old_train + 1)
    else:
        # adjust weights for covariate shift
        with torch.no_grad():
            imp_weights = ratio_estimator(train_inputs)  # (*q_batch_shape, num_total)
        imp_weights /= imp_weights.sum(dim=-1, keepdim=True)
        imp_weights = imp_weights.view(*q_batch_shape, 1, num_total, 1)

    cum_weights = (rank_mask * imp_weights).sum(
        -2
    )  # (*q_batch_shape, grid_size, q_batch_size)
    conf_pred_mask = torch.sigmoid((cum_weights - alpha) / temp)

    return conf_pred_mask, updated_gps, q_conf_scores


def assess_coverage(model, inputs, targets, alpha=0.05, temp=1e-6):
    targets = targets.squeeze(-1)
    model.eval()
    model.standard()

    with torch.no_grad():
        y_post = model.posterior(inputs, observation_noise=True)
        y_mean = y_post.mean
        y_std = y_post.variance.sqrt()
        std_scale = stats.norm.ppf(1 - alpha / 2.0)
        cred_lb = y_mean - std_scale * y_std
        cred_ub = y_mean + std_scale * y_std
        cred_lb = cred_lb.squeeze()
        cred_ub = cred_ub.squeeze()

        std_coverage = (
            (targets > cred_lb) * (targets < cred_ub)
        ).float().sum() / targets.shape[0]

        _, _, _, conf_lb, conf_ub = construct_conformal_bands(model, inputs, alpha, temp)
        conformal_coverage = (
            (targets > conf_lb.to(targets)) * (targets < conf_ub.to(targets))
        ).float().sum() / targets.shape[0]

    model.standard()

    return std_coverage.item(), conformal_coverage.item()


class ConformalPosterior(Posterior):
    def __init__(
        self,
        X,
        gp,
        target_bounds,
        alpha,
        tgt_grid_res,
        ratio_estimator=None,
        temp=1e-2,
        max_grid_refinements=4,
    ):
        self.gp = gp
        self.X = X
        self.target_bounds = target_bounds
        self.tgt_grid_res = tgt_grid_res
        self.alpha = alpha
        self.ratio_estimator = ratio_estimator
        self.temp = temp
        self.max_grid_refinements = max_grid_refinements

    @property
    def device(self):
        return self.X.shape

    @property
    def dtype(self):
        return self.X.shape

    @property
    def event_shape(self):
        return self.X.shape[:-2] + torch.Size([1])

    def rsample(self, sample_shape=(), base_samples=None):
        target_grid, conf_pred_mask, conditioned_gps, _, _ = construct_conformal_bands(
            self.gp, self.X, self.alpha, self.temp
        )
        self.gp.conf_tgt_grid = target_grid
        self.gp.conf_pred_mask = conf_pred_mask

        conditioned_gps.standard()
        reshaped_x = self.X[:, None].expand(-1, target_grid.size(-3), -1, -1)
        posteriors = conditioned_gps.posterior(reshaped_x)

        out = posteriors.rsample(sample_shape, base_samples)
        return out


class PassSampler(MCSampler):
    def __init__(self, num_samples):
        super().__init__(batch_range=(0, -2))
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = True

    def _construct_base_samples(self, posterior, shape):
        pass


class ConformalSingleTaskGP(SingleTaskGP):
    def __init__(
        self,
        conformal_bounds,
        alpha,
        tgt_grid_res,
        ratio_estimator=None,
        max_grid_refinements=4,
        temp=1e-2,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conformal_bounds = conformal_bounds
        self.alpha = alpha
        self.is_conformal = False
        self.tgt_grid_res = tgt_grid_res
        self.ratio_estimator = ratio_estimator
        self.max_grid_refinements = max_grid_refinements
        self.temp = temp

    def conformal(self):
        self.is_conformal = True

    def standard(self):
        self.is_conformal = False

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        if self.is_conformal:
            posterior = ConformalPosterior(
                X,
                self,
                self.conformal_bounds,
                alpha=self.alpha,
                tgt_grid_res=self.tgt_grid_res,
                ratio_estimator=self.ratio_estimator,
                max_grid_refinements=self.max_grid_refinements,
            )
            if hasattr(self, "outcome_transform"):
                posterior = self.outcome_transform.untransform_posterior(posterior)
            return posterior
        else:
            return super().posterior(
                X=X,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
            )

    @property
    def batch_shape(self):
        if self.is_conformal:
            try:
                return self.conf_pred_mask.shape[:-2]
            except:
                pass
        return self.train_inputs[0].shape[:-2]
