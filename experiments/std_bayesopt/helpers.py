import numpy as np
import torch

from scipy import stats

from gpytorch.distributions import MultivariateNormal

from botorch.sampling import SobolQMCNormalSampler
from botorch.posteriors import GPyTorchPosterior

import torchsort


def generate_target_grid(grid_center, grid_scale, grid_res, tail_prob):
    """
    Create dense pointwise target grid
    Args:
        grid_center: (*q_batch_shape, q_batch_size, target_dim)
        grid_scale: (*q_batch_shape, q_batch_size, target_dim)
        grid_res: int
        tail_prob: float
    Returns:
        target_grid: (*q_batch_shape, grid_res, q_batch_size, target_dim)
    """
    # TODO check target_dim > 1 case
    cred_levels = np.linspace(tail_prob, 1 - tail_prob, grid_res)
    std_factors = stats.norm.ppf(cred_levels)[:, None, None]  # (grid_res, 1, 1)
    std_factors = torch.from_numpy(std_factors).to(grid_center)
    target_grid = grid_center.unsqueeze(-3) + std_factors * grid_scale.unsqueeze(-3)
    return target_grid


def conf_mask_to_bounds(target_grid, conf_pred_mask):
    """
    Convert dense pointwise target grid and corresponding conformal mask
    to pointwise conformal prediction intervals.
    Args:
        target_grid: (*q_batch_shape, grid_res, q_batch_size, target_dim)
        conf_pred_mask: (*q_batch_shape, grid_res, q_batch_size, target_dim)
    Returns:
        conf_lb: (*q_batch_shape, q_batch_size, target_dim)
        conf_ub: (*q_batch_shape, q_batch_size, target_dim)
    """
    out_shape = target_grid.shape[:-3] + target_grid.shape[-2:]
    # target_dim = target_grid.size(-1)
    flat_tgt_grid = target_grid.flatten(0, -4)
    flat_pred_mask = conf_pred_mask.flatten(0, -4)
    conf_ub = (
        torch.stack(
            [
                grid[mask >= 0.5].max(0)[0]
                for grid, mask in zip(flat_tgt_grid, flat_pred_mask)
            ]
        )
    )
    conf_lb = (
        torch.stack(
            [
                grid[mask >= 0.5].min(0)[0]
                for grid, mask in zip(flat_tgt_grid, flat_pred_mask)
            ]
        )
    )
    conf_lb = conf_lb.view(*out_shape)
    conf_ub = conf_ub.view(*out_shape)

    return conf_lb, conf_ub


def sample_grid_points(grid_lb, grid_ub, num_samples):
    """
    Convert pointwise conformal prediction intervals to uniformly
    sampled grid points
    Args:
        grid_lb: (*q_batch_shape, q_batch_size, target_dim)
        grid_ub: (*q_batch_shape, q_batch_size, target_dim)
        num_samples: int
    Returns:
        grid_samples: (*q_batch_shape, num_samples, q_batch_size, target_dim)
    """
    q_batch_shape = grid_lb.shape[:-2]
    q_batch_size = grid_lb.size(-2)
    grid_range = grid_ub - grid_lb
    unif_samples = torch.rand(
        *q_batch_shape, num_samples, q_batch_size, 1
    ).to(grid_lb)
    grid_samples = grid_lb.unsqueeze(-3) + unif_samples * grid_range.unsqueeze(-3)
    return grid_samples


def construct_conformal_bands(model, inputs, alpha, temp, grid_res, max_grid_refinements, ratio_estimator=None):
    """
    Construct dense pointwise target grid and conformal prediction mask
    Args:
        model: gpytorch.models.GP
        inputs: (*q_batch_shape, q_batch_size, input_dim)
        alpha: float
        temp: float
        grid_res: int
    Returns:
        target_grid: (*q_batch_shape, q_batch_size, grid_res, 1, target_dim)
        conf_pred_mask: (*q_batch_shape, q_batch_size, grid_res, 1, target_dim)
        conditioned_models: gpytorch.models.GP
    """

    shape_msg = "inputs should have shape (*q_batch_shape, q_batch_size, input_dim)" \
                f"instead has shape {inputs.shape}"
    assert inputs.ndim >= 3, shape_msg
    assert grid_res >= 2, "grid resolution must be at least 2"
    q_batch_size = inputs.size(-2)

    # dummy q-batch dimension
    # inputs = inputs.unsqueeze(-2)

    # initialize target grid with pointwise credible sets
    model.eval()
    with torch.no_grad():
        y_post = model.posterior(inputs, observation_noise=True)
        # y_mean = y_post.mean
        # y_std = y_post.variance.sqrt()

    # setup
    # grid_center = y_mean
    # grid_scale = y_std
    conditioned_models = None
    for refine_step in range(0, max_grid_refinements + 1):
        # construct target grid, conformal prediction mask
        # target_grid = generate_target_grid(
        #     grid_center, grid_scale, grid_res, tail_prob=alpha / 4.
        # )

        sampler = SobolQMCNormalSampler(grid_res)
        target_grid = sampler(y_post)
        assert target_grid.size(0) == grid_res
        target_grid = torch.movedim(target_grid, 0, -3)

        conf_pred_mask, conditioned_models, q_conf_scores = conformal_gp_regression(
            model, inputs, target_grid, alpha, temp=temp
        )
        # TODO revisit for target_dim > 1
        # reshape to (*q_batch_shape, q_batch_size, grid_res, 1, target_dim)
        conf_pred_mask = conf_pred_mask.unsqueeze(-1)  # create target dim
        q_conf_scores = q_conf_scores.unsqueeze(-1)  # create target dim

        num_accepted = (conf_pred_mask >= 0.5).float().sum(-3)

        # center target grid around best scoring point in current grid
        with torch.no_grad():
            avg_score = q_conf_scores.mean(dim=-2, keepdim=True)
            avg_score_argmax = avg_score.argmax(
                dim=-3, keepdim=True
            )  # (*q_batch_shape, 1, 1, 1)
            avg_score_argmax = avg_score_argmax.expand(
                *[-1] * (avg_score.ndim - 2), q_batch_size, -1
            )  # (*q_batch_shape, 1, q_batch_size, 1)
            new_center = torch.gather(
                target_grid, dim=-3, index=avg_score_argmax
            )
            new_center = new_center.squeeze(
                -3
            )  # (*q_batch_shape, q_batch_size, 1, target_dim)

        # TODO revisit when target_dim > 1, replace squeeze with movedim
        new_center = new_center.squeeze(-1)
        covar_scale = (1. + num_accepted.float() / grid_res).pow(2).squeeze(-1)
        # covariance matrix has extra last dimension (*batch_shape, n, n)
        covar_scale = covar_scale.unsqueeze(-1)

        grid_center = y_post.mvn.mean

        # import pdb;
        # pdb.set_trace()

        assert grid_center.size() == new_center.size()
        recenter_grid = False
        # if not torch.allclose(grid_center, new_center, rtol=1e-2):
        #     recenter_grid = True
        #     grid_center = new_center.contiguous()

        # if fewer than two points have been accepted, increase grid resolution
        # too_few_accepted = ()
        refine_grid = False
        if torch.any(num_accepted < 2):
            refine_grid = True
            grid_res *= 2

        # if too many/all grid elements are in prediction set, expand the grid bounds
        # grid_lb_accepted = (conf_pred_mask[..., 0, :, :] >= 0.5)
        # grid_ub_accepted = (conf_pred_mask[..., -1, :, :] >= 0.5)
        # too_many_accepted = (num_accepted > grid_res - 2)
        # should_expand = grid_lb_accepted + grid_ub_accepted + too_many_accepted

        # warning! evaluating bc of weird GPyTorch lazy tensor broadcasting bug
        grid_covar = y_post.mvn.lazy_covariance_matrix.evaluate().contiguous()
        expand_grid = False
        if torch.any(num_accepted > grid_res - 2):
            grid_covar = (covar_scale * grid_covar)
            expand_grid = True
            # assert grid_scale.size() == should_expand.size()
            # grid_scale += (grid_scale * num_accepted.float() / grid_res)

        if recenter_grid or expand_grid:
            y_post = GPyTorchPosterior(
                MultivariateNormal(grid_center, grid_covar)
            )

        if not any([recenter_grid, refine_grid, expand_grid]):
            break

        if refine_step < max_grid_refinements:
            del conditioned_models

    # print(
    #     f"conformal step: {refine_step},",
    #     f"grid resolution: {grid_res},",
    #     f"tail_prob: {cred_tail_prob:0.4f},",
    #     f"accepted ratio: {num_accepted.mean().item():0.2f}"
    # )

    # TODO improve error handling for case when max_iter is exhausted
    # if this error is raised, it's a good indication of a bug somewhere else
    if torch.any(num_accepted < 2):
        import pdb; pdb.set_trace()
        raise RuntimeError("not enough grid points accepted")

    return target_grid, conf_pred_mask, conditioned_models


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
    grid_size, q_batch_size, target_dim = target_grid.shape[-3:]
    q_batch_shape = test_inputs.shape[:-2]

    gp.eval()
    # gp.conf_pred_mask = (
    #     None  # without this line the deepcopy in `gp.condition_on_observations` fails
    # )

    # retraining: condition the GP at every target grid point for every test input
    expanded_inputs = test_inputs.unsqueeze(-3).expand(
        *[-1] * len(q_batch_shape), target_grid.shape[-3], -1, -1
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
    train_labels = train_labels.view(*target_grid.shape[:-2], num_total, target_dim)

    if hasattr(updated_gps, "input_transform"):
        train_inputs = updated_gps.input_transform.untransform(train_inputs)
    if hasattr(updated_gps, "output_transform"):
        train_labels = updated_gps.output_transform.untransform(train_labels)

    # compute conformal scores (pointwise posterior predictive log-likelihood)
    # TODO extend to target_dim > 1 case
    if target_dim > 1:
        raise NotImplementedError

    posterior = updated_gps.posterior(train_inputs, observation_noise=True)
    pred_dist = torch.distributions.Normal(posterior.mean, posterior.variance.sqrt())
    conf_scores = pred_dist.log_prob(train_labels)
    conf_scores = conf_scores.view(*q_batch_shape, grid_size, num_total)
    q_conf_scores = conf_scores[..., num_old_train:].detach()

    original_shape = conf_scores.shape
    ranks_by_score = torchsort.soft_rank(
        conf_scores.flatten(0, -2),
        regularization="l2",
        regularization_strength=temp,
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

    # num_accepted = (conf_pred_mask >= 0.5).float().sum(-2)
    # if torch.any(num_accepted < 2):
    #     import pdb; pdb.set_trace()

    return conf_pred_mask, updated_gps, q_conf_scores


def assess_coverage(
        model, inputs, targets, alpha, temp, grid_res, max_grid_refinements, ratio_estimator=None
):
    """
    Args:
        model (gpytorch.models.GP)
        inputs: (batch_size, input_dim)
        targets: (batch_size, target_dim)
        alpha: float
        temp: float
    Returns:
        std_coverage: float
        conformal_coverage: float
    """
    model.eval()
    with torch.no_grad():
        y_post = model.posterior(inputs, observation_noise=True)
        y_mean = y_post.mean
        y_std = y_post.variance.sqrt()
        std_scale = stats.norm.ppf(1 - alpha / 2.0)
        cred_lb = y_mean - std_scale * y_std
        cred_ub = y_mean + std_scale * y_std

        std_coverage = (
            (targets > cred_lb) * (targets < cred_ub)
        ).float().mean()

        target_grid, conf_pred_mask, _ = construct_conformal_bands(
            model, inputs[:, None], alpha, temp, grid_res,
            max_grid_refinements, ratio_estimator
        )
        conf_lb, conf_ub = conf_mask_to_bounds(target_grid, conf_pred_mask)
        conf_lb = conf_lb.squeeze(-2).to(targets)
        conf_ub = conf_ub.squeeze(-2).to(targets)

        conformal_coverage = (
            (targets > conf_lb) * (targets < conf_ub)
        ).float().mean()

    return std_coverage.item(), conformal_coverage.item()
