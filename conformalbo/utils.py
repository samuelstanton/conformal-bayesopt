import torch
import argparse
import numpy as np
import math
import pandas as pd

from collections.__init__ import namedtuple

from scipy.stats import norm, spearmanr

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.test_functions import Branin, Levy, Ackley, Michalewicz
from botorch.test_functions.multi_objective import (
    BraninCurrin,
    Penicillin,
    CarSideImpact,
    ZDT2,
)
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list

from conformalbo.optim.optimize import optimize_acqf_sgld, optimize_acqf_sgld_list
from conformalbo.helpers import assess_coverage


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--problem", type=str, default="branin")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n_batch", type=int, default=50)
    parser.add_argument("--num_init", type=int, default=10)
    parser.add_argument("--mc_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--min_alpha", type=float, default=0.05)
    parser.add_argument("--method", type=str, default="exact")
    parser.add_argument("--tgt_grid_res", type=int, default=64)
    parser.add_argument("--temp", type=float, default=1e-2)
    parser.add_argument("--max_grid_refinements", type=int, default=4)
    parser.add_argument("--sgld_steps", type=int, default=100)
    parser.add_argument("--sgld_temperature", type=float, default=1e-3)
    parser.add_argument("--sgld_lr", type=float, default=1e-3)
    parser.add_argument("--rand_orthant", action="store_true")
    return parser.parse_args()


def sample_random_orthant(base_samples):
    rand_mask = (torch.rand(base_samples.size(-1)) < 0.5).to(base_samples)
    base_samples = 0.5 * rand_mask + 0.5 * base_samples
    return base_samples


def generate_initial_data(n, fn, NOISE_SE, device, dtype, is_poisson=False, use_sobol_eng=True,
                          rand_orthant=False):
    """
    Generate random training data
    """
    sobol_failed = False
    if use_sobol_eng:
        try:
            sobol_eng = torch.quasirandom.SobolEngine(dimension=fn.dim, scramble=True)
            base_samples = sobol_eng.draw(n).to(device=device, dtype=dtype)
        except:
            sobol_failed = True
            pass

    if not use_sobol_eng or sobol_failed:
        base_samples = torch.rand(
            n, fn.dim, device=device, dtype=dtype
        )

    if rand_orthant:
        train_x = sample_random_orthant(base_samples)
    else:
        train_x = base_samples

    # transform inputs before passing to fn
    bounds = fn.bounds.to(base_samples)
    cube_loc = bounds[0]
    cube_scale = (bounds[1] - bounds[0])
    exact_obj = fn(train_x * cube_scale + cube_loc)
    if exact_obj.ndim == 1:
        exact_obj = exact_obj.unsqueeze(-1)  # add output dimension if we need to
    best_observed_value = exact_obj.max().item()

    # add noise to observed labels
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    return train_x, train_obj, best_observed_value


def initialize_noise_se(fn, noise_se, device, dtype):
    # we initialize the noise se to [noise_se * the standard deviation of 5k
    # random points ]
    
    # initializes noise standard erorrs
    rand_x = torch.rand(5000, fn.dim, device=device, dtype=dtype)
    
    cube_loc = fn.bounds[0]
    cube_scale = fn.bounds[1] - fn.bounds[0]
    
    exact_obj = fn(rand_x * cube_scale + cube_loc)    
    if exact_obj.ndim == 1:
        exact_obj = exact_obj.unsqueeze(-1) # add output dimension if we need to
    
    output_std = exact_obj.std(0)
    return noise_se * output_std

def initialize_model(
    train_x,
    train_obj,
    train_yvar=None,
    state_dict=None,
    method="variational",
    loss="elbo",
    **kwargs
):
    # TODO: enable multitaskGP here
    transform = Standardize(train_obj.shape[-1]).to(train_x.device)
    t_train_obj = transform(train_obj)[0]
    # define models for objective and constraint
    if method == "variational":
        model_obj = get_var_model(
            train_x, t_train_obj, train_yvar, is_poisson=False, **kwargs
        )
        if loss == "elbo":
            mll = VariationalELBO(
                model_obj.likelihood, model_obj, num_data=train_x.shape[-2]
            )
        elif loss == "pll":
            mll = PredictiveLogLikelihood(
                model_obj.likelihood, model_obj, num_data=train_x.shape[-2]
            )
    elif method == "exact":
        model_obj = get_exact_model(train_x, t_train_obj, train_yvar, **kwargs)
        mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)

    model_obj = model_obj.to(train_x.device)
    # load state dict if it is passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj, transform


def update_random_observations(
    BATCH_SIZE, best_random, bounds, problem=lambda x: x, dim=6, noise_se=0.1,
):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = torch.rand(BATCH_SIZE, dim).to(bounds) * (bounds[1] - bounds[0]) + bounds[0]
    rand_f = problem(rand_x)
    # return true fn value
    next_random_best = rand_f.max().item()
    # rand_y = rand_f + noise_se * torch.randn_like(rand_f)
    best_random.append(max(best_random[-1], next_random_best))
    return best_random


def get_exact_model(
    x,
    y,
    yvar,
    **kwargs
):
    # TODO: setup noise constraint in batched setting
    model = SingleTaskGP(
        train_X=x,
        train_Y=y,
        likelihood=GaussianLikelihood(
            noise_constraint=Interval(1e-4, 5e-1),
            batch_shape = torch.Size((y.shape[-1],)) if y.shape[-1] > 1 else torch.Size(),
        ),
        outcome_transform=None,
        input_transform=None,
    )

    if yvar is not None:
        model.likelihood.raw_noise.detach_()
        model.likelihood.noise = yvar
    return model


def get_problem(problem, dim, num_objectives=1):
    if problem == "levy":
        return Levy(dim=dim, negate=True)
    elif problem == "branin":
        return Branin(negate=True)
    elif problem == "ackley":
        return Ackley(dim=dim, negate=True)
    elif problem == "michal":
        return Michalewicz(dim=dim, negate=True)
    elif problem == "branincurrin":
        return BraninCurrin(negate=True)
    elif problem == "zdt2":
        # TODO: check if we need to negate
        return ZDT2(dim=dim, num_objectives=2, negate=True)
    elif problem == "penicillin":
        # TODO: check if we need to negate
        return Penicillin(negate=True)
    elif problem == "carside":
        return CarSideImpact(negate=True)
    # we may want to try ToyRobust as well for more of a noise robustness
    # argument


def optimize_acqf_and_get_observation(
    acq_func,
    bounds,
    fn,
    BATCH_SIZE=1,
    outcome_constraint=None,
    noise_se=0.0,
    is_list=False,
    NUM_RESTARTS=5,
    RAW_SAMPLES=128,
    is_poisson=False,
    sequential=False,
    **kwargs,
):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""

    _kwargs = kwargs # og kwargs
    kwargs = {
        "acq_function_list" if is_list else "acq_function": acq_func,
        "bounds": bounds,
        "q": BATCH_SIZE,
        "num_restarts": NUM_RESTARTS,
        "raw_samples": RAW_SAMPLES,
        "options": {"batch_limit": 5, "maxiter": 200, "sample_around_best": False},
        "sequential": sequential,
        "return_best_only": False
    }

    if hasattr(acq_func, 'ratio_estimator') and acq_func.ratio_estimator is not None:
        optimizer = optimize_acqf_sgld
        kwargs['options']['callback'] = acq_func.ratio_estimator.optimize_callback
        kwargs = {**kwargs, **_kwargs}
    elif is_list and hasattr(acq_func[0], 'ratio_estimator') and acq_func[0].ratio_estimator is not None:
        optimizer = optimize_acqf_sgld_list
        kwargs = {**kwargs, **_kwargs}
        sequential = True
        kwargs.pop("q")
        kwargs.pop("sequential")
    elif is_list:
        optimizer = optimize_acqf_list
        sequential = True
        kwargs.pop("q")
        kwargs.pop("sequential")
        kwargs.pop("return_best_only")
    else:
        optimizer = optimize_acqf

    # optimize
    candidates, acq_vals = optimizer(**kwargs)
    # reshape
    all_x = candidates.view(-1, BATCH_SIZE, bounds.size(-1)).detach()
    # accommodate sequentially optimized batches
    all_acq_vals = acq_vals.detach().view(all_x.size(0), -1).sum(-1)

    # rescale candidates before passing to obj fn
    cube_loc = fn.bounds[0].to(all_x)
    cube_scale = (fn.bounds[1] - fn.bounds[0]).to(all_x)
    all_f = fn(all_x * cube_scale + cube_loc)
    all_f = all_f.view(all_x.size(0), BATCH_SIZE, -1)
    all_y = all_f + noise_se * torch.randn_like(all_f)

    best_idx = all_acq_vals.argmax()
    best_x = all_x[best_idx]
    best_y = all_y[best_idx]
    best_f = all_f[best_idx]
    best_a = all_acq_vals[best_idx]

    # print(f"acq fn opt: best {best_a.item():0.4f}, worst {all_acq_vals.min():0.4f}")
    # print(f"x*: {best_x}")

    return best_x, best_y, best_f, best_a, all_x, all_y


fields = ("inputs", "targets")
defaults = (np.array([]), np.array([]))
DataSplit = namedtuple("DataSplit", fields, defaults=defaults)


def update_splits(
    train_split: DataSplit,
    val_split: DataSplit,
    test_split: DataSplit,
    new_split: DataSplit,
    holdout_ratio: float = 0.2,
):
    r"""
    This utility function updates train, validation and test data splits with
    new observations while preventing leakage from train back to val or test.
    New observations are allocated proportionally to prevent the
    distribution of the splits from drifting apart.

    New rows are added to the validation and test splits randomly according to
    a binomial distribution determined by the holdout ratio. This allows all splits
    to be updated with as few new points as desired. In the long run the split proportions
    will converge to the correct values.
    """
    train_inputs, train_targets = train_split
    val_inputs, val_targets = val_split
    test_inputs, test_targets = test_split

    # shuffle new data
    new_inputs, new_targets = new_split
    new_perm = np.random.permutation(
        np.arange(new_inputs.shape[0])
    )
    new_inputs = new_inputs[new_perm]
    new_targets = new_targets[new_perm]

    unseen_inputs = safe_np_cat([test_inputs, new_inputs])
    unseen_targets = safe_np_cat([test_targets, new_targets])

    num_rows = train_inputs.shape[0] + val_inputs.shape[0] + unseen_inputs.shape[0]
    num_test = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        unseen_inputs.shape[0],
    )
    num_test = max(test_inputs.shape[0], num_test) if test_inputs.size else max(1, num_test)

    # first allocate to test split
    test_split = DataSplit(unseen_inputs[:num_test], unseen_targets[:num_test])

    resid_inputs = unseen_inputs[num_test:]
    resid_targets = unseen_targets[num_test:]
    resid_inputs = safe_np_cat([val_inputs, resid_inputs])
    resid_targets = safe_np_cat([val_targets, resid_targets])

    # then allocate to val split
    num_val = min(
        np.random.binomial(num_rows, holdout_ratio / 2.),
        resid_inputs.shape[0],
    )
    num_val = max(val_inputs.shape[0], num_val) if val_inputs.size else max(1, num_val)
    val_split = DataSplit(resid_inputs[:num_val], resid_targets[:num_val])

    # train split gets whatever is left
    last_inputs = resid_inputs[num_val:]
    last_targets = resid_targets[num_val:]
    train_inputs = safe_np_cat([train_inputs, last_inputs])
    train_targets = safe_np_cat([train_targets, last_targets])
    train_split = DataSplit(train_inputs, train_targets)

    return train_split, val_split, test_split


def safe_np_cat(arrays, **kwargs):
    if all([arr.size == 0 for arr in arrays]):
        return np.array([])
    cat_arrays = [arr for arr in arrays if arr.size]
    return np.concatenate(cat_arrays, **kwargs)


def fit_and_transform(transform, train_arr, holdout_arr=None):
    transform.train()
    train_result = transform(train_arr)
    transform.eval()
    if holdout_arr is None:
        return train_result
    return train_result, transform(holdout_arr)


def fit_surrogate(train_X, train_Y):
    surrogate = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    surrogate_mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
    surrogate.train()
    surrogate.requires_grad_(True)
    fit_gpytorch_model(surrogate_mll)
    surrogate.requires_grad_(False)
    surrogate.eval()
    return surrogate


def set_alpha(cfg, num_train):
    alpha = 1 / math.sqrt(num_train)
    cfg.conformal_params['alpha'] = alpha
    return alpha


def set_beta(cfg):
    beta = norm.ppf(1 - cfg.conformal_params.alpha / 2.).item()
    # import pdb; pdb.set_trace()
    if 'beta' in cfg.acquisition:
        cfg.acquisition.beta = beta
    return beta


def display_metrics(metrics):
    df = pd.DataFrame((metrics,)).round(4)
    print(df.to_markdown())


def evaluate_surrogate(cfg, surrogate, holdout_X, holdout_Y, dr_estimator=None, log_prefix=''):
    eval_metrics = {}
    # p(y | x, D)
    y_post = surrogate.posterior(holdout_X, observation_noise=True)
    # NLL, RMSE, and Spearman's Rho
    eval_metrics['nll'] = -1 * torch.distributions.Normal(y_post.mean, y_post.variance.sqrt()).log_prob(holdout_Y).mean().item()
    eval_metrics["rmse"] = (y_post.mean - holdout_Y).pow(2).mean().sqrt().item()
    try:
        s_rho = np.stack([
            spearmanr(
                holdout_Y[:, idx], y_post.mean[:, idx]
            ).correlation for idx in range(holdout_Y.shape[-1])
        ]).mean().item()
    except Exception:
        s_rho = float('NaN')
    eval_metrics["s_rho"] = s_rho
    # credible and conformal coverage
    eval_metrics["exp_cvrg"] = 1 - cfg.conformal_params.alpha
    eval_metrics["cred_cvrg"], eval_metrics["conf_cvrg"] = assess_coverage(
        surrogate, holdout_X, holdout_Y, ratio_estimator=dr_estimator, **cfg.conformal_params
    )
    # add log prefix
    if len(log_prefix) > 0:
        eval_metrics = {'_'.join([log_prefix, key]): val for key, val in eval_metrics.items()}
    display_metrics(eval_metrics)
    return eval_metrics
