import torch
import argparse
import numpy as np

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.test_functions import Branin, Levy, Ackley
from botorch.test_functions.multi_objective import (
    BraninCurrin,
    Penicillin,
    CarSideImpact,
    ZDT2,
)
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list

try:
    from ratio_estimation import optimize_acqf_sgld, optimize_acqf_sgld_list
except ImportError:
    from conformalbo.ratio_estimation import optimize_acqf_sgld, optimize_acqf_sgld_list


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
    rand_y = problem(rand_x)
    # return true fn value
    next_random_best = rand_y.max().item()
    # rand_y += noise_se * torch.randn_like(rand_y)
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
