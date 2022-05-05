import numpy as np
import torch
import warnings

from scipy import stats

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch import lazify


from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
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
    original_out_shape = target_grid.shape[:-3] + target_grid.shape[-2:]
    
    target_grid = target_grid.movedim(-1, 0)
    conf_pred_mask = conf_pred_mask.movedim(-1, 0)
    
    # offset by 1d here from the original one
    new_out_shape = target_grid.shape[:-2] + target_grid.shape[-1:]
    
    # this flattens any structured q batches
    flat_tgt_grid = target_grid.flatten(0, -3)
    flat_pred_mask = conf_pred_mask.flatten(0, -3)
    
    # now flatten across the target dim
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
    conf_lb = conf_lb.view(*new_out_shape)
    conf_ub = conf_ub.view(*new_out_shape)
    
    # but this out shape doesn't align with our expected out shape
    # so we move the target dim back
    conf_lb = conf_lb.movedim(0, -1)
    conf_ub = conf_ub.movedim(0, -1)
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


def construct_conformal_bands(model, inputs, alpha, temp, grid_res, max_grid_refinements, sampler,
                              ratio_estimator=None):
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

    # setup
    # grid_center = y_mean
    # grid_scale = y_std
    conditioned_models = None
    best_result = (-1, None, None, None, None)
    best_refine_step = 0
    for refine_step in range(0, max_grid_refinements + 1):
        # construct target grid, conformal prediction mask

        # sampler = SobolQMCNormalSampler(grid_res)
        sampler._sample_shape = torch.Size([grid_res])
        target_grid = sampler(y_post)

        # TODO: check these lines for shaping errors
        weights = y_post.mvn.log_prob(target_grid.squeeze(-1))
        weights = weights[..., None, None]

        assert target_grid.size(0) == grid_res
        target_grid = torch.movedim(target_grid, 0, -3)
        weights = torch.movedim(weights, 0, -3)

        conf_pred_mask, conditioned_models, q_conf_scores = conformal_gp_regression(
            model, inputs, target_grid, alpha, temp=temp
        )
        target_dim = target_grid.shape[-1]
        if target_dim == 1:
            # reshape to (*q_batch_shape, grid_res, q_batch_size, target_dim)
             conf_pred_mask = conf_pred_mask.unsqueeze(-1)  # create target dim
             q_conf_scores = q_conf_scores.unsqueeze(-1)  # create target dim

        num_accepted = (conf_pred_mask >= 0.5).float().sum(-3)
        min_accepted = num_accepted.min()
        max_accepted = num_accepted.max()
        if min_accepted > best_result[0]:
            best_result = (
                min_accepted, target_grid, weights, conf_pred_mask, conditioned_models
            )
            best_refine_step = refine_step

        grid_center = y_post.mvn.mean

        recenter_grid = False

        accept_ratio = (num_accepted.float() / grid_res)
        diff = accept_ratio - 0.5
        scale_exp = 2. / q_batch_size  # volume is exponential in the batch size
        covar_scale = 1. + diff.sign() * diff.abs().pow(scale_exp)
        covar_scale = covar_scale.squeeze(-1)  # drop target dim

        # covariance matrix has extra last dimension (*batch_shape, n, n)
        if target_dim == 1:
            covar_scale = covar_scale.unsqueeze(-1)

        # warning! evaluating bc of weird GPyTorch lazy tensor broadcasting bug
        grid_covar = y_post.mvn.lazy_covariance_matrix.evaluate().contiguous()
        rescale_grid = False
        if min_accepted < 2 or max_accepted > grid_res - 2:
            # batched mt case
            if covar_scale.shape[-1] != grid_covar.shape[-1]:
                covar_scale = covar_scale.reshape(*grid_covar.shape[:-1], 1)
            grid_covar = (covar_scale * grid_covar)
            rescale_grid = True

        refine_grid = False

        if recenter_grid or rescale_grid:
            if type(y_post.mvn) is MultivariateNormal:
                init_fn = MultivariateNormal
                kwargs = {}
            elif type(y_post.mvn) is MultitaskMultivariateNormal:
                init_fn = MultitaskMultivariateNormal
                kwargs = {"interleaved": y_post.mvn._interleaved}
            y_post = GPyTorchPosterior(
                init_fn(grid_center, lazify(grid_covar), **kwargs)
            )
        if not any([recenter_grid, refine_grid, rescale_grid]):
            break

        if refine_step < max_grid_refinements:
            conditioned_models.train() # don't let this build up
            del conditioned_models
            torch.cuda.empty_cache()

    min_accepted, target_grid, weights, conf_pred_mask, conditioned_models = best_result
    num_accepted = (conf_pred_mask >= 0.5).float().sum(-3)

    # TODO improve error handling for case when max_iter is exhausted
    # if this error is raised, it's a good indication of a bug somewhere else
    min_accepted = int(min_accepted)
    max_accepted = int(num_accepted.max())
    msg = f"\ntarget_grid: {target_grid.shape}, {min_accepted} - {max_accepted} grid points accepted"
    if torch.any(num_accepted < 2):
            warnings.warn(msg)

    return target_grid, weights, conf_pred_mask, conditioned_models


def conformal_gp_regression(
    gp, test_inputs, target_grid, alpha=0.2, temp=1e-6, ratio_estimator=None, **kwargs
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
    if gp._aug_batch_shape != torch.Size([]):
        # we need to put the batch shape dim first for the posterior call
        train_inputs = train_inputs.movedim(len(q_batch_shape) + 1, 0)
        train_inputs = train_inputs[0] # damn botorch batch dims
        
    train_labels = updated_gps.prediction_strategy.train_labels
    train_labels = train_labels.view(*target_grid.shape[:-2], num_total, target_dim)

    if hasattr(updated_gps, "input_transform"):
        train_inputs = updated_gps.input_transform.untransform(train_inputs)
    if hasattr(updated_gps, "output_transform"):
        train_labels = updated_gps.output_transform.untransform(train_labels)

    # compute conformal scores (pointwise posterior predictive log-likelihood)
    posterior = updated_gps.posterior(train_inputs, observation_noise=True)
    post_var = est_train_post_var(updated_gps, observation_noise=True, target_dim=target_dim)
    pred_dist = torch.distributions.Normal(posterior.mean, post_var.sqrt())
    
    conf_scores = pred_dist.log_prob(train_labels)
    if target_dim == 1:
        conf_scores = conf_scores.view(*q_batch_shape, grid_size, num_total)
    else:
        conf_scores = conf_scores.view(*q_batch_shape, grid_size, num_total, target_dim)
        conf_scores = conf_scores.transpose(-1, -2) # now q x grid x target x n
        
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

    # TODO: check this summation index
    cum_weights = (rank_mask * imp_weights).sum(
        -2
    )  # (*q_batch_shape, grid_size, q_batch_size)
    conf_pred_mask = torch.sigmoid((cum_weights - alpha) / temp)

    if target_dim > 1:
        conf_pred_mask = conf_pred_mask.transpose(-1, -2)
        q_conf_scores = q_conf_scores.transpose(-1, -2)

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
    print("at beginning of cov.: ", torch.cuda.memory_allocated() / 1024**3)
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

        grid_sampler = SobolQMCNormalSampler(grid_res, resample=True)
        # grid_sampler = IIDNormalSampler(grid_res, resample=True, collapse_batch_dims=False)
        target_grid, _, conf_pred_mask, _ = construct_conformal_bands(
            model, inputs[:, None], alpha, temp, grid_res,
            max_grid_refinements, grid_sampler, ratio_estimator
        )
        try:
            # remove this targets when done!!
            conf_lb, conf_ub = conf_mask_to_bounds(target_grid, conf_pred_mask)
            conf_lb = conf_lb.squeeze(-2).to(targets)
            conf_ub = conf_ub.squeeze(-2).to(targets)

            conformal_coverage = (
                (targets > conf_lb) * (targets < conf_ub)
            ).float().mean()
        except:
            print("Warning heldout set coverage evaluation failed. Returning nan")
            conformal_coverage = torch.tensor(float('nan'))
            
    model.train()
    del target_grid, conf_pred_mask
    torch.cuda.empty_cache()
    print("at end: ", torch.cuda.memory_allocated() / 1024**3)

    return std_coverage.item(), conformal_coverage.item()


def est_train_post_var(model, observation_noise=False, num_samples=8, target_dim=1):
    """
    Estimate Var[f | D] pointwise on train.
    If `observation_noise` is `True`, return Var[y | D].
    https://www-users.cse.umn.edu/~saad/PDF/umsi-2005-082.pdf
    """
    noise = model.likelihood.noise
    kxx_plus_noise = model.prediction_strategy.lik_train_train_covar
    kxx = kxx_plus_noise.lazy_tensors[0]
    z = (torch.rand(kxx.shape[-1], num_samples) > 0.5).to(noise)
    z = 2. * (z - 0.5)  # if z in {0, 1} then denom can be 0 if n is small
    denom = (z * z).sum(-1) + 1e-6
    approx_eigs = (z * kxx.matmul(kxx_plus_noise.inv_matmul(z))).sum(-1) / denom
    if observation_noise:
        approx_eigs += 1.
    approx_post_var = (noise * approx_eigs)
    
    # TODO: check some of the batching
    if target_dim == 1:
        approx_post_var = approx_post_var.unsqueeze(-1)  # create target dim
    else:
        approx_post_var = approx_post_var.transpose(-1, -2)
    return approx_post_var
