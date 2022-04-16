from helpers import PassSampler
import torch
import time

# from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
# from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler

from utils import (
    generate_initial_data,
    initialize_model,
    parse,
    optimize_acqf_and_get_observation,
    update_random_observations,
    get_problem,
    assess_coverage,
)
from helpers import qConformalExpectedImprovement, qConformalNoisyExpectedImprovement

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
    bounds = bb_fn.bounds

    best_observed_ei, best_observed_nei, best_observed_cei, best_observed_cnei, best_random = (
        [],
        [],
        [],
        [],
        [],
    )
    coverage_ei, coverage_nei, coverage_cei, coverage_cnei = (
        [],
        [],
        [],
        [],
    )

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

    mll_ei, model_ei = initialize_model(
        train_x_ei, train_obj_ei, train_yvar, method=method, alpha=alpha,
    )

    train_x_nei, train_obj_nei = train_x_ei, train_obj_ei
    best_observed_value_nei = best_observed_value_ei
    mll_nei, model_nei = initialize_model(
        train_x_nei, train_obj_nei, train_yvar, method=method, alpha=alpha,
    )

    train_x_cei, train_obj_cei = train_x_ei, train_obj_ei
    best_observed_value_cei = best_observed_value_ei
    mll_cei, model_cei = initialize_model(
        train_x_cei, train_obj_cei, train_yvar, method=method, alpha=alpha, tgt_grid_res=tgt_grid_res,
    )

    train_x_cnei, train_obj_cnei = train_x_ei, train_obj_ei
    best_observed_value_cnei = best_observed_value_ei
    mll_cnei, model_cnei = initialize_model(
        train_x_cnei, train_obj_cnei, train_yvar, method=method, alpha=alpha, tgt_grid_res=tgt_grid_res,
    )

    best_observed_ei.append(best_observed_value_ei)
    best_observed_nei.append(best_observed_value_nei)
    best_observed_cei.append(best_observed_value_cei)
    best_observed_cnei.append(best_observed_value_cnei)
    best_random.append(best_observed_value_ei)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, n_batch + 1):

        t0 = time.time()

        # fit the models
        fit_gpytorch_model(mll_ei)
        fit_gpytorch_model(mll_nei)
        fit_gpytorch_model(mll_cei)
        fit_gpytorch_model(mll_cnei)

        # first we assess coverage on the heldout set (which is random)
        coverage_ei.append(assess_coverage(model_ei, heldout_x, heldout_obj, alpha))
        coverage_nei.append(assess_coverage(model_nei, heldout_x, heldout_obj, alpha))
        coverage_cei.append(assess_coverage(model_cei, heldout_x, heldout_obj, alpha))
        coverage_cnei.append(assess_coverage(model_cnei, heldout_x, heldout_obj, alpha))
        
        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model_ei,
            best_f=(train_obj_ei).max(),
            sampler=qmc_sampler,
        )

        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
        )

        model_cei.conformal()
        qconEI = qConformalExpectedImprovement(
            model=model_cei,
            best_f=(train_obj_cei).max(),
            sampler=PassSampler(mc_samples),
        )
        qconEI.objective._verify_output_shape = False

        model_cnei.conformal()
        qconNEI = qConformalNoisyExpectedImprovement(
            model=model_cnei,
            X_baseline=train_x_cnei,
            sampler=PassSampler(mc_samples),
            cache_root=False,
        )
        qconNEI.objective._verify_output_shape = False

        # optimize and get new observation
        optimize_acqf_kwargs = {
            "bounds": bb_fn.bounds,
            "BATCH_SIZE": batch_size,
            "fn": bb_fn,
            "noise_se": noise_se,
        }
        new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(
            qEI, **optimize_acqf_kwargs
        )
        new_x_nei, new_obj_nei = optimize_acqf_and_get_observation(
            qNEI, **optimize_acqf_kwargs
        )
        new_x_cei, new_obj_cei = optimize_acqf_and_get_observation(
            qconEI, **optimize_acqf_kwargs
        )
        new_x_cnei, new_obj_cnei = optimize_acqf_and_get_observation(
            qconNEI, **optimize_acqf_kwargs
        )

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])

        train_x_cei = torch.cat([train_x_cei, new_x_cei])
        train_obj_cei = torch.cat([train_obj_cei, new_obj_cei])

        train_x_cnei = torch.cat([train_x_cnei, new_x_cnei])
        train_obj_cnei = torch.cat([train_obj_cnei, new_obj_cnei])

        # update progress
        best_random = update_random_observations(batch_size, best_random, bounds, bb_fn, dim=bounds.shape[1])
        best_value_ei = train_obj_ei.max().item()
        best_value_nei = train_obj_nei.max().item()
        best_value_cei = train_obj_cei.max().item()
        best_value_cnei = train_obj_cnei.max().item()
        best_observed_ei.append(best_value_ei)
        best_observed_nei.append(best_value_nei)
        best_observed_cei.append(best_value_cei)
        best_observed_cnei.append(best_value_cnei)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei,
            train_obj_ei,
            train_yvar,
            method=method,
            # model_ei.state_dict(),
        )
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            train_yvar,
            # model_nei.state_dict(),
            method=method,
        )
        mll_cei, model_cei = initialize_model(
            train_x_cei,
            train_obj_cei,
            train_yvar,
            method=method,
            # model_ei.state_dict(),
        )
        mll_cnei, model_cnei = initialize_model(
            train_x_cnei,
            train_obj_cnei,
            train_yvar,
            # model_nei.state_dict(),
            method=method,
        )
        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI, qconEI, qconNEI) = "
                f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}, {best_value_cei:>4.2f}, {best_value_cnei:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
            print("coverage", coverage_ei[-1], coverage_cnei[-1],)
        else:
            print(".", end="")

    output_dict = {
        "best_achieved":{
            "rnd": best_random,
            "ei": best_observed_ei,
            "nei": best_observed_nei,
            "cei": best_observed_cei,
            "cnei": best_observed_cnei,
        },
        "coverage": {
            "ei": coverage_ei,
            "nei": coverage_nei,
            "cei": coverage_cei,
            "cnei": coverage_cnei,            
        },
        "inputs": {
            "ei": train_x_ei,
            "nei": train_x_nei,
            "cei": train_x_cei,
            "cnei": train_x_cnei,
        }
    }
    return output_dict

if __name__ == "__main__":
    args = parse()
    output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
