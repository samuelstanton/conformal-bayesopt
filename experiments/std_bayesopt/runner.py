from helpers import PassSampler
import torch
import time

# from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler

from utils import (
    generate_initial_data,
    initialize_model,
    parse,
    optimize_acqf_and_get_observation,
    update_random_observations,
    get_problem,
)
from experiments.std_bayesopt.helpers import assess_coverage
from helpers import qConformalExpectedImprovement, qConformalNoisyExpectedImprovement
from botorch.models.transforms import Standardize, Normalize

def main(
    seed: int = 0,
    dim: int = 10,
    method: str = "exact",
    batch_size: int = 3,
    n_batch: int = 50,
    tgt_grid_res: int = 64,
    mc_samples: int = 256,
    num_init: int = 10,
    noise_se: float = 0.1,
    dtype: str = "double",
    verbose: bool = True,
    output: str = None,
    problem: str = None,
    alpha: float = 0.05,
):
    dtype = torch.double if dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.random.manual_seed(seed)

    bb_fn = get_problem(problem, dim)
    bb_fn = bb_fn.to(device, dtype)
    # bounds = torch.stack((torch.zeros(bb_fn.dim), torch.ones(bb_fn.dim)).to(device, dtype)
    bounds = torch.zeros_like(bb_fn.bounds)
    bounds[1] += 1.

    keys = ["rnd", "ei", "nei", "kg", "cei"]
    best_observed = {k: [] for k in keys}
    coverage = {k: [] for k in keys}

    train_yvar = torch.tensor(noise_se ** 2, device=device, dtype=dtype)

    # call helper functions to generate initial training data and initialize model
    (
        train_x_ei,
        train_obj_ei,
        best_observed_value_ei,
    ) = generate_initial_data(
        num_init, bb_fn, noise_se, device, dtype
    )
    heldout_x, heldout_obj, _ = generate_initial_data(10 * num_init, bb_fn, noise_se, device, dtype)

    mll_model_dict = {}
    data_dict = {}
    for k in keys:
        mll_and_model = initialize_model(
            train_x_ei, train_obj_ei, train_yvar,
            method=method, alpha=alpha, tgt_grid_res=tgt_grid_res,
        )
        mll_model_dict[k] = (mll_and_model)
        best_observed[k].append(best_observed_value_ei)
        data_dict[k] = (train_x_ei, train_obj_ei)

    optimize_acqf_kwargs = {
        "bounds": bounds,
        "BATCH_SIZE": batch_size,
        "fn": bb_fn,
        "noise_se": noise_se,
    }

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, n_batch + 1):
        t0 = time.time()
        for k in keys:

            if k == "rnd":
                # update random
                best_observed[k] = update_random_observations(batch_size, best_observed[k], bb_fn.bounds, bb_fn, dim=bounds.shape[1])
                continue

            # fit the model
            mll, model, trans = mll_model_dict[k]
            inputs, objective = data_dict[k]
            trans.eval()
            t_objective = trans(objective)[0]
            model.requires_grad_(True)
            fit_gpytorch_model(mll)
            model.requires_grad_(False)

            # now assess coverage on the heldout set
            # TODO: update the heldout sets
            coverage[k].append(assess_coverage(model, heldout_x, trans(heldout_obj)[0], alpha))
            print(coverage[k][-1], k)

            # now prepare the acquisition
            qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)
            # qmc_sampler = IIDNormalSampler(num_samples=mc_samples)
            if k == "ei":
                acqf = qExpectedImprovement(
                    model=model,
                    best_f=(t_objective).max(),
                   sampler=qmc_sampler,
                )
            elif k == "nei":
                acqf = qNoisyExpectedImprovement(
                    model=model,
                    X_baseline=inputs,
                    sampler=qmc_sampler,
                )
            elif k == "kg":
                acqf = qKnowledgeGradient(
                    model=model,
                    current_value=t_objective.max(),
                    num_fantasies=None,
                    sampler=qmc_sampler,
                )
            elif k == "cei":
                model.conformal()
                acqf = qConformalExpectedImprovement(
                    model=model,
                    best_f=(t_objective).max(),
                    sampler=PassSampler(mc_samples),
                )
                acqf.objective._verify_output_shape = False
            elif k == "cnei":
                model.conformal()
                acqf = qConformalNoisyExpectedImprovement(
                    model=model,
                    X_baseline=inputs,
                    sampler=PassSampler(mc_samples),
                    cache_root=False,
                )
                acqf.objective._verify_output_shape = False
        
            # optimize acquisition
            new_x, new_obj = optimize_acqf_and_get_observation(
                acqf, **optimize_acqf_kwargs
            )
            print(objective.max(), new_obj, k)

            inputs = torch.cat([inputs, new_x])
            objective = torch.cat([objective, new_obj])

            best_observed[k].append(objective.max().item())
            # prepare new model
            mll, model, trans = initialize_model(
                inputs,
                objective,
                method=method,
            )
            mll_model_dict[k] = (mll, model, trans)
            data_dict[k] = inputs, objective

        t1 = time.time()

        if verbose:
            best_random = best_observed["rnd"][-1]
            best_ei = best_observed["ei"][-1]
            best_nei = best_observed["nei"][-1]
            best_cei = best_observed["cei"][-1]
            best_cnei = best_observed.get("cnei", [-float("inf")])[-1]
            best_kg = best_observed["kg"][-1]
            print(
                f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI, qconEI, qconNEI, qKG) = "
                f"({best_random:>4.2f}, {best_ei:>4.2f}, {best_nei:>4.2f}, {best_cei:>4.2f}, {best_cnei:>4.2f}), "
                f"{best_kg:>4.2f}), time = {t1-t0:>4.2f}.",
                end="",
            )
            # print("coverage", coverage)
        else:
            print(".", end="")

    output_dict = {
        "best_achieved": best_observed,
        "coverage": coverage,
        "inputs": {k: data_dict[k][0] for k in keys},
    }
    return output_dict


if __name__ == "__main__":
    args = parse()
    output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
