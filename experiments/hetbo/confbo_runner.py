import matplotlib.pyplot as plt
import torch
import botorch
import gpytorch

from botorch.models import HeteroskedasticSingleTaskGP
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.penalized import PenalizedAcquisitionFunction
from botorch.optim.optimize import optimize_acqf

import math

import sys
sys.path.append("..")
from conformalbo.acquisitions import qConformalExpectedImprovement, qConformalUpperConfidenceBound
from botorch.sampling import IIDNormalSampler
from conformalbo.ratio_estimation import RatioEstimator, optimize_acqf_sgld
from botorch.models import SingleTaskGP
from lambo.utils import DataSplit, update_splits

import argparse

bounds = torch.tensor([-10, 10.])
fn = lambda x: (10. * torch.sin(x)+1) * (torch.sin(3. * x) / x).nan_to_num(1.)
noise = lambda x: 2.0 * torch.sigmoid(0.5 * x)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--acqf", type=str, default="cucb")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()

def generate_data_for_gp(inp=None, k = 3, n = 15):
    if inp is None:
        inp = torch.rand(n, 1) * (bounds[1] - bounds[0]) + bounds[0]
        inp = inp[inp.abs() > 3]
        inp = inp.view(-1,1)
    full_y = torch.stack([fn(inp) + noise(inp).sqrt() * torch.randn_like(inp) for _ in range(k)])
    return inp, full_y.mean(0), full_y.std(0)

def run_conformal_experiment(
    n_steps=80, n_init=10, acqf="cucb",
    mc_samples=64, min_alpha=0.05, temp=0.01,
    device=0,
):
    bounds = torch.tensor([-10, 10.])
    fn = lambda x: (10. * torch.sin(x)+1) * (torch.sin(3. * x) / x).nan_to_num(1.)
    noise = lambda x: 0.5 * torch.sigmoid(0.5 * x)

    device = torch.device("cuda:"+str(device) if torch.cuda.is_available() else "cpu")

    x, y, _ = generate_data_for_gp(k=1, n=n_init)
    x = x.to(device)
    y = y.to(device)
    # for step in range(n_steps):
    step = 0
    while x.shape[0] < (n_steps + n_init):

        if step % 5 == 0:
            print(step, y.max())

        model = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        try:
            with gpytorch.settings.cholesky_max_tries(10):
                fit_gpytorch_scipy(mll);
        except:
            print("Warning gp fitting failed")
        model.requires_grad_(False)

        sampler = IIDNormalSampler(num_samples=mc_samples, batch_range=(0, -3))

        num_total = x.shape[0]

        # prepare density ratio estimator
        rx_estimator = RatioEstimator(x.size(-1), x.device, x.dtype)
        rx_estimator.dataset._update_splits(
            DataSplit(x.cpu(), torch.zeros(x.size(0), 1))
        )

        # set alpha
        alpha = max(1.0 / math.sqrt(num_total), min_alpha)
        conformal_kwargs = {}
        conformal_kwargs['alpha'] = alpha
        conformal_kwargs['temp'] = temp
        conformal_kwargs['max_grid_refinements'] = 0
        conformal_kwargs['ratio_estimator'] = rx_estimator
        conformal_kwargs["grid_res"] = 64

        if acqf == "cucb":
            acqf = qConformalUpperConfidenceBound(
                model=model, sampler=sampler, beta=0.2,
                **conformal_kwargs
            )
        elif acqf == "cei":
            acqf = qConformalExpectedImprovement(model=model, best_f=y.max(),
                                                 sampler=sampler, **conformal_kwargs)

        candidates, _ = optimize_acqf_sgld(
            acq_function=acqf,
            bounds=bounds.view(2,1).to(device),
            q=1,
            num_restarts = 5,
            raw_samples=256,
            sgld_steps=200,
            sgld_temperature=0.05,
            sgld_lr=1e-3,
        )
        next_x = candidates.view(-1,1)
        _, next_y, _ = generate_data_for_gp(next_x, k=1)
        x = torch.cat((x, next_x.to(device)))
        y = torch.cat((y, next_y.to(device)))
        # y_std = torch.cat((y_std, next_y_std))

        del model

        step += 1

    return x, y, None

if __name__ == "__main__":
    args = parse()
    torch.random.manual_seed(args.seed)
    x, y, _ = run_conformal_experiment(acqf=args.acqf, device=args.device)
    torch.save({"x": x, "y": y}, args.output)
