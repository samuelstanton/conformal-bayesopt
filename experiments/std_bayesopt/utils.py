import torch
import argparse

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize, Normalize

from botorch.test_functions import Branin, Levy, Ackley
from botorch.optim import optimize_acqf

from helpers import PassSampler, ConformalSingleTaskGP, generate_target_grid, conformal_gp_regression

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
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--method", type=str, default="exact")
    return parser.parse_args()

def generate_initial_data(
    n, fn, NOISE_SE, device, dtype, is_poisson=False
):
    # generate training data
    train_x = torch.rand(n, fn.dim, device=device, dtype=dtype) * (fn.bounds[1] - fn.bounds[0]) + fn.bounds[0]
    exact_obj = fn(train_x).unsqueeze(-1)  # add output dimension
    # exact_con = outcome_constraint(train_x).unsqueeze(-1)  # add output dimension
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    best_observed_value = exact_obj.max().item()
    return train_x, train_obj, best_observed_value

def initialize_model(
    train_x,
    train_obj,
    train_yvar=None,
    state_dict=None,
    method="variational",
    loss="elbo",
    **kwargs
):
    # define models for objective and constraint
    if method == "variational":
        model_obj = get_var_model(
            train_x, train_obj, train_yvar, is_poisson=False, **kwargs
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
        model_obj = get_exact_model(train_x, train_obj, train_yvar, **kwargs)
        mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)

    model_obj = model_obj.to(train_x.device)
    # load state dict if it is passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj

def update_random_observations(BATCH_SIZE, best_random, problem=lambda x: x, dim=6):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = torch.rand(BATCH_SIZE, dim)
    next_random_best = problem(rand_x).max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random

def get_exact_model(
    x, y, yvar, use_input_transform=True, use_outcome_transform=True, alpha=0.05,
        tgt_grid_res=32, **kwargs
):
    conformal_bounds = torch.tensor([[-3., 3.]]).t() # this can be standardized w/o worry?

    model = ConformalSingleTaskGP(
        train_X=x,
        train_Y=y,
        likelihood=GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2))
        if yvar is None
        else None,
        outcome_transform=Standardize(y.shape[-1]) if use_outcome_transform else None,
        input_transform=Normalize(x.shape[-1]) if use_input_transform else None,
        alpha=alpha,
        conformal_bounds=conformal_bounds,
        tgt_grid_res=tgt_grid_res
    ).to(x)
    if yvar is not None:
        model.likelihood.raw_noise.detach_()
        model.likelihood.noise = yvar.item()
    return model

def get_problem(problem, dim):
    if problem == "levy":
        return Levy(dim=dim, negate=True)
    elif problem == "branin":
        return Branin(negate=True)
    elif problem == "ackley":
        return Ackley(dim=dim, negate=True)

def assess_coverage(model, inputs, targets, alpha = 0.05):
    with torch.no_grad():
        model.standard()
        # TODO: fix coverage for alpha
        conf_region = model.posterior(inputs, observation_noise=True).mvn.confidence_region()

        std_coverage = ((targets > conf_region[0]) * (targets < conf_region[1])).float().sum() / targets.shape[0]

        # convert conformal prediction mask to prediction set
        model.conformal()
        target_grid = generate_target_grid(model.conformal_bounds, model.tgt_grid_res).to(inputs)
        conf_pred_mask, _ = conformal_gp_regression(model, inputs, target_grid, alpha, temp=1e-4)

        conformal_conf_region = construct_conformal_bands(conf_pred_mask, target_grid)
        conformal_conf_region = [cc.to(targets) for cc in conformal_conf_region]
        conformal_coverage = (
            (targets > conformal_conf_region[0]) * (targets < conformal_conf_region[1])
        ).float().sum() / targets.shape[0]
    model.train()
    model.standard()
    return std_coverage, conformal_coverage

def construct_conformal_bands(conf_pred_mask, target_grid):
    masked_targets = conf_pred_mask.to(target_grid).unsqueeze(-1) * target_grid
    conf_ub = masked_targets.max(-2)[0].cpu().view(-1)
    conf_lb = -1 * (-masked_targets).max(-2)[0].cpu().view(-1)
    return conf_ub, conf_lb

def optimize_acqf_and_get_observation(
    acq_func,
    bounds,
    BATCH_SIZE,
    fn,
    outcome_constraint=None,
    noise_se=0.0,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    is_poisson=False,
    sequential=False,
):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=sequential,
    )
    # observe new values
    new_x = candidates.detach()
    exact_obj = fn(new_x).unsqueeze(-1)  # add output dimension

    new_obj = exact_obj + noise_se * torch.randn_like(exact_obj)
    return new_x, new_obj
