import numpy as np
import torch
import warnings

from scipy import stats

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch import lazify


from botorch.sampling import IIDNormalSampler
from botorch.posteriors import GPyTorchPosterior

import torchsort


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

    # lets do this slow first
    # we iterate over the traget dims
    dim_lb_list, dim_ub_list = [], []
    for dim_target_grid, dim_conf_pred_mask in zip(target_grid, conf_pred_mask):
        # we assume only q = 1
        flat_tgt_grid = dim_target_grid.squeeze(-1) 
        flat_pred_mask = dim_conf_pred_mask.squeeze(-1)
        
        # now we sort the grid indices
        sorted_flat_tgt_grid, sorted_flat_inds = torch.sort(flat_tgt_grid, -1, descending=False)
        sorted_pred_mask = torch.stack([ff[ii] for ff, ii in zip(flat_pred_mask, sorted_flat_inds)])
        
        # iterate over the sorted 
        lb_list, ub_list = [], []
        for grid, mask in zip(sorted_flat_tgt_grid, sorted_pred_mask):
            inds = torch.where(mask >= 0.5)[0]
            
            if len(inds) == 0:
                tsr_nan = torch.ones([1]).to(mask) / 0.
                lb_list.append(tsr_nan)
                ub_list.append(tsr_nan)
            else:
                
                # select first and last indices
                grid_ub = inds[-1]
                grid_lb = inds[0]

                # now we interpolate, we want the x where y == 0.5
                if grid_lb != 0:
                    const_lower_term = (0.5 - mask[grid_lb - 1]) * (grid[grid_lb] - grid[grid_lb - 1])\
                        / (mask[grid_lb] - mask[grid_lb - 1])
                    lower_bound = grid[grid_lb - 1] + const_lower_term
                else:
                    lower_bound = grid[grid_lb]

                if grid_ub != mask.shape[0] - 1:
                    const_upper_term = (mask[grid_ub] - 0.5) * (grid[grid_ub + 1] - grid[grid_ub])\
                        / (mask[grid_ub] - mask[grid_ub + 1])
                    upper_bound = grid[grid_ub] + const_upper_term
                else:
                    upper_bound = grid[grid_ub]
                
                lb_list.append(lower_bound.view(-1))
                ub_list.append(upper_bound.view(-1))
        dim_lb_list.append(torch.stack(lb_list))
        dim_ub_list.append(torch.stack(ub_list))

    # offset by 1d here from the original one
    new_out_shape = target_grid.shape[:-2] + target_grid.shape[-1:]
    
    conf_lb = torch.stack(dim_lb_list)
    conf_ub = torch.stack(dim_ub_list)

    conf_lb = conf_lb.view(*new_out_shape)
    conf_ub = conf_ub.view(*new_out_shape)
    
    # but this out shape doesn't align with our expected out shape
    # so we move the target dim back
    conf_lb = conf_lb.movedim(0, -1)
    conf_ub = conf_ub.movedim(0, -1)
    return conf_lb, conf_ub


def construct_conformal_bands(model, inputs, alpha, temp, grid_res, max_grid_refinements, sampler,
                              ratio_estimator=None, mask_ood=True, randomized=False):
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

    # initialize target grid with pointwise credible sets
    model.eval()
    with torch.no_grad():
        y_post = model.posterior(inputs, observation_noise=True)
        grid_center = y_post.mvn.mean
        # warning! evaluating bc of weird GPyTorch lazy tensor broadcasting bug
        grid_covar = y_post.mvn.lazy_covariance_matrix.evaluate().contiguous()
        y_post = refine_grid_dist(y_post, grid_center, 2. * grid_covar)

    # setup
    conditioned_models = None
    best_result = (float('inf'), None, None, None, None, None)
    best_refine_step = 0
    for refine_step in range(0, max_grid_refinements + 1):
        # construct target grid, conformal prediction mask

        sampler._sample_shape = torch.Size([grid_res])
        target_grid = sampler(y_post)

        # TODO: check these lines for shaping errors
        # (*q_batch_shape, grid_size)
        weights = y_post.mvn.log_prob(target_grid.squeeze(-1))
        # (*q_batch_shape, grid_size, 1, 1)
        weights = weights[..., None, None]

        assert target_grid.size(0) == grid_res
        target_grid = torch.movedim(target_grid, 0, -3)
        weights = torch.movedim(weights, 0, -3)

        conf_pred_mask, conditioned_models, q_conf_scores, ood_mask = conformal_gp_regression(
            model, inputs, target_grid, alpha, temp=temp, ratio_estimator=ratio_estimator,
            mask_ood=mask_ood, randomized=randomized
        )

        num_accepted = (conf_pred_mask >= 0.5).float().sum(-3)
        min_accepted = num_accepted.min()
        max_accepted = num_accepted.max()
        accept_ratio = (num_accepted.float() / grid_res)
        diff = accept_ratio - 0.5
        grid_score = diff.pow(2).mean()
        if grid_score < best_result[0]:
            best_result = (
                grid_score, target_grid, weights, conf_pred_mask, conditioned_models, ood_mask
            )
            best_refine_step = refine_step

        grid_center = y_post.mvn.mean
        # warning! evaluating bc of weird GPyTorch lazy tensor broadcasting bug
        grid_covar = y_post.mvn.lazy_covariance_matrix.evaluate().contiguous()

        scale_exp = 2. / q_batch_size  # volume is exponential in the batch size
        covar_scale = 1. + diff.sign() * diff.abs().pow(scale_exp)
        # reshape as (*batch_shape, n, 1)
        # in both batched mt and single task cases
        covar_scale = covar_scale.reshape(*grid_covar.shape[:-1], 1)
        
        refine_grid = False

        recenter_grid = False

        rescale_grid = False
        if min_accepted < 2 or max_accepted > grid_res - 2:
            grid_covar = covar_scale * grid_covar
            rescale_grid = True

        if recenter_grid or rescale_grid:
            y_post = refine_grid_dist(y_post, grid_center, grid_covar)
        if not any([recenter_grid, refine_grid, rescale_grid]):
            break

        if refine_step < max_grid_refinements:
            conditioned_models.train() # delete pred caches before deletion.
            del conditioned_models
            torch.cuda.empty_cache()

    min_accepted, target_grid, weights, conf_pred_mask, conditioned_models, ood_mask = best_result
    num_accepted = (conf_pred_mask >= 0.5).float().sum(-3)

    # TODO improve error handling for case when max_iter is exhausted
    # if this error is raised, it's a good indication of a bug somewhere else
    min_accepted = int(min_accepted)
    max_accepted = int(num_accepted.max())
    msg = f"\ntarget_grid: {target_grid.shape}, {min_accepted} - {max_accepted} grid points accepted"
    # if torch.any(num_accepted < 2):
    #         warnings.warn(msg)

    return target_grid, weights, conf_pred_mask, conditioned_models, ood_mask


def refine_grid_dist(y_post, grid_center, grid_covar):
    if type(y_post.mvn) is MultivariateNormal:
        init_fn = MultivariateNormal
        kwargs = {}
    elif type(y_post.mvn) is MultitaskMultivariateNormal:
        init_fn = MultitaskMultivariateNormal
        kwargs = {"interleaved": y_post.mvn._interleaved}
    y_post = GPyTorchPosterior(
        init_fn(grid_center, lazify(grid_covar), **kwargs)
    )
    return y_post


def conformal_gp_regression(
    gp, test_inputs, target_grid, alpha=0.2, temp=1e-6, ratio_estimator=None, mask_ood=True,
    randomized=False, **kwargs
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
        train_inputs = train_inputs[0]  # damn botorch batch dims
        
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
    conf_scores = conf_scores.view(*q_batch_shape, grid_size, num_total, target_dim)
    conf_scores = conf_scores.transpose(-1, -2)  # (*q_batch_shape, grid_size, target_dim, num_total)

    threshold = conf_scores[..., num_old_train:]
    # rank_mask.shape = (*q_batch_shape, grid_size, target_dim, num_total, q_batch_size)
    rank_mask = (threshold.unsqueeze(-2) >= conf_scores.unsqueeze(-1)).to(conf_scores)
    if temp > 0:
        mask_sum = rank_mask.sum(dim=-2, keepdim=True)
        rank_mask = 1. - torch.sigmoid(
            (conf_scores.unsqueeze(-1) - threshold.unsqueeze(-2)) / temp
        )
        rank_mask *= mask_sum / rank_mask.sum(dim=-2, keepdim=True)

    # get importance weights to adjust for covariate shift
    imp_weights = torch.zeros_like(rank_mask, requires_grad=False)
    if ratio_estimator is None:
        imp_weights[..., :num_old_train, :] = 1.
        imp_weights[
            ..., np.arange(-q_batch_size, 0), np.arange(-q_batch_size, 0)
        ] = 1.
    else:
        dr_old_inputs = ratio_estimator(train_inputs[..., :num_old_train, :]).clamp_min(1e-6)
        dr_new_inputs = ratio_estimator(train_inputs[..., num_old_train:, :]).clamp_min(1e-6)
        imp_weights[..., :num_old_train, :] = dr_old_inputs[..., None, :, None]
        imp_weights[
            ..., np.arange(-q_batch_size, 0), np.arange(-q_batch_size, 0)
        ] = dr_new_inputs[..., None, :]
    imp_weights /= imp_weights.sum(dim=-2, keepdim=True)

    # sum masked importance weights, soft Heaviside again
    masked_weights = rank_mask * imp_weights
    cum_weights = masked_weights.cumsum(
        -2
    )  # (*q_batch_shape, grid_size, target_dim, num_total, q_batch_size)

    if randomized:
        diff = masked_weights[..., -1, :].clamp_min(1e-6)
        lb_prob = (
            (cum_weights[..., -1, :] - alpha) / diff
        ).clamp(0., 1.)
        randomization_mask = torch.distributions.Bernoulli(probs=1. - lb_prob).sample()
        cum_weights = cum_weights[..., -2, :] + randomization_mask.to(diff) * diff
    else:
        cum_weights = cum_weights[..., -1, :]

    if temp > 0:
        conf_pred_mask = torch.sigmoid((cum_weights - alpha) / temp)
    else:
        conf_pred_mask = (cum_weights > alpha).to(cum_weights)

    # (acquisition only) apply soft mask to OOD inputs where any target value is accepted
    if temp > 0:
        ood_mask = torch.sigmoid((imp_weights[..., num_old_train:, :].sum(-2) - alpha) / temp)
    else:
        ood_mask = (imp_weights[..., num_old_train:, :].sum(-2) >= alpha)

    # reshape to (*q_batch_shape, grid_size, q_batch_size, target_dim)
    conf_pred_mask = conf_pred_mask.transpose(-1, -2)
    conf_scores = conf_scores.transpose(-1, -2)
    q_conf_scores = conf_scores[..., num_old_train:, :].detach()

    return conf_pred_mask, updated_gps, q_conf_scores, ood_mask


def assess_coverage(
        model, inputs, targets, alpha, temp, grid_res, max_grid_refinements, ratio_estimator=None, randomized=False
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
    # print("at beginning of cov.: ", torch.cuda.memory_allocated() / 1024**3)
    model.eval()
    with torch.no_grad():
        y_post = model.posterior(inputs, observation_noise=True)
        y_mean = y_post.mean
        y_std = y_post.variance.sqrt()
        std_scale = stats.norm.ppf(1 - alpha / 2.0)
        cred_lb = y_mean - std_scale * y_std
        cred_ub = y_mean + std_scale * y_std

        std_coverage = (
            (targets >= cred_lb) * (targets <= cred_ub)
        ).float().mean()

        grid_sampler = IIDNormalSampler(grid_res, resample=False, collapse_batch_dims=False)
        target_grid, _, conf_pred_mask, _, _ = construct_conformal_bands(
            model, inputs[:, None], alpha, temp, grid_res,
            max_grid_refinements, grid_sampler, ratio_estimator, mask_ood=False, randomized=randomized
        )
        try:
            conf_lb, conf_ub = conf_mask_to_bounds(target_grid, conf_pred_mask)
            conf_lb = conf_lb.squeeze(-2).to(targets)
            conf_ub = conf_ub.squeeze(-2).to(targets)

            # nanmean returns all nans if all values are nan
            conformal_coverage = (
                (targets >= conf_lb) * (targets <= conf_ub)
            ).float().nanmean()
        except:
            print("Warning heldout set coverage evaluation failed. Returning nan")
            conformal_coverage = torch.tensor(float('nan'))
            
    model.train()
    del target_grid, conf_pred_mask
    torch.cuda.empty_cache()
    # print("at end: ", torch.cuda.memory_allocated() / 1024**3)

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
