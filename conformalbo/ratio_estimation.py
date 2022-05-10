import torch
from torch import nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from experiments.std_bayesopt.optim import SGLD

from lambo.utils import DataSplit, update_splits


def optimize_acqf_sgld(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Callable]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    sgld_steps: int = 100,
    lr: float = 1e-2,
    temperature: float = .1,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    @NOTE: This is a modified version of `botorch.optim.optimize_acqf`.

    Intended SGLD stationary distribution is $p(x) \propto \exp\{ - a(x) \}$

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        nonlinear_inequality_constraints: A list of callables with that represent
            non-linear inequality constraints of the form `callable(x) >= 0`. Each
            callable is expected to take a `(num_restarts) x q x d`-dim tensor as an
            input and return a `(num_restarts) x q`-dim tensor with the constraint
            values. The constraints will later be passed to SLSQP. You need to pass in
            `batch_initial_conditions` in this case. Using non-linear inequality
            constraints also requires that `batch_limit` is set to 1, which will be
            done automatically if not specified in `options`.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        kwargs: Additonal keyword arguments.

    Returns:
        A two-element tuple containing

        - a `(num_restarts) x q x d`-dim tensor of generated candidates.
        - a tensor of associated acquisition values. If `sequential=False`,
            this is a `(num_restarts)`-dim tensor of joint acquisition values
            (with explicit restart dimension if `return_best_only=False`). If
            `sequential=True`, this is a `q`-dim tensor of expected acquisition
            values conditional on having observed canidates `0,1,...,i-1`.

    Example:
        >>> # generate `q=2` candidates jointly using 20 random restarts
        >>> # and 512 raw samples
        >>> candidates, acq_value = optimize_acqf(qEI, bounds, 2, 20, 512)

        >>> generate `q=3` candidates sequentially using 15 random restarts
        >>> # and 256 raw samples
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> candidates, acq_value_list = optimize_acqf(
        >>>     qEI, bounds, 3, 15, 256, sequential=True
        >>> )
    """
    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError(
                "`return_best_only=False` only supported for joint optimization."
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf_sgld(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=True,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}

    # Handle the trivial case when all features are fixed
    if fixed_features is not None and len(fixed_features) == bounds.shape[-1]:
        X = torch.tensor(
            [fixed_features[i] for i in range(bounds.shape[-1])],
            device=bounds.device,
            dtype=bounds.dtype,
        )
        X = X.expand(q, *X.shape)
        with torch.no_grad():
            acq_value = acq_function(X)
        return X, acq_value

    if batch_initial_conditions is None:
        if nonlinear_inequality_constraints:
            raise NotImplementedError(
                "`batch_initial_conditions` must be given if there are non-linear "
                "inequality constraints."
            )
        if raw_samples is None:
            raise ValueError(
                "Must specify `raw_samples` when `batch_initial_conditions` is `None`."
            )

        ic_gen = (
            gen_one_shot_kg_initial_conditions
            if isinstance(acq_function, qKnowledgeGradient)
            else gen_batch_initial_conditions
        )
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    batch_limit: int = options.get(
        "batch_limit", num_restarts if not nonlinear_inequality_constraints else 1
    )
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    batched_ics = batch_initial_conditions.split(batch_limit)
    for i, batched_ics_ in enumerate(batched_ics):
        batched_ics_ = batched_ics_.requires_grad_(True)
        sgld = SGLD([batched_ics_], lr=lr, momentum=.9, temperature=temperature)
        
        for _s in range(sgld_steps):
            sgld.zero_grad()

            batched_acq_values_ = acq_function(batched_ics_)

            batched_acq_values_.sum().backward()

            if callable(options.get('callback')):
                options.get('callback')(batched_acq_values_.detach().cpu())
            
            sgld.step()
        
        with torch.no_grad():
            batch_candidates_curr, batch_acq_values_curr = batched_ics_.detach(), acq_function(batched_ics_.detach())

        # assert not batch_candidates_curr.requires_grad

        # optimize using random restart optimization
        # batch_candidates_curr, batch_acq_values_curr = gen_candidates_scipy(
        #     initial_conditions=batched_ics_,
        #     acquisition_function=acq_function,
        #     lower_bounds=bounds[0],
        #     upper_bounds=bounds[1],
        #     options={k: v for k, v in options.items() if k not in INIT_OPTION_KEYS},
        #     inequality_constraints=inequality_constraints,
        #     equality_constraints=equality_constraints,
        #     nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        #     fixed_features=fixed_features,
        # )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values


class RatioEstimator(nn.Module):
    class _Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self._prior_ratio = 1
            self.cls_train_split, self.cls_val_split, self.cls_test_split = update_splits(
                train_split=DataSplit(),
                val_split=DataSplit(),
                test_split=DataSplit(),
                new_split=DataSplit(),
                holdout_ratio=0.2,
            )

        def __len__(self):
            return len(self.cls_train_split.inputs)

        def __getitem__(self, index):
            return self.cls_train_split.inputs[index], self.cls_train_split.targets[index]

        @property
        def emp_prior(self):
            return self._prior_ratio

        @property
        def num_positive(self):
            if len(self) == 0:
                return 0
            _, train_targets = self.cls_train_split
            return train_targets.sum().item()

        def recompute_emp_prior(self):
            n = len(self)
            n_p = self.num_positive
            if n_p > 0 and n - n_p > 0:
                self._prior_ratio = n / n_p - 1.0
            return self._prior_ratio

        def _update_splits(self, new_split):
            self.cls_train_split, self.cls_val_split, self.cls_test_split = update_splits(
                train_split=self.cls_train_split,
                val_split=self.cls_val_split,
                test_split=self.cls_test_split,
                new_split=new_split,
                holdout_ratio=0.2,
            )
            self.recompute_emp_prior()

    def __init__(self, in_size=1, device=None):
        super().__init__()

        self.device = device
        self.dataset = RatioEstimator._Dataset()

        ## Remains uniform, when untrained.
        self.classifier = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_size, 1),
        # )
        # for p in self.classifier.parameters():
        #     p.data.fill_(0)
        #     p.requires_grad_(True)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(
            self.classifier.parameters(), lr=1e-2, betas=(0.0, 1e-2)
        )

    @torch.no_grad()
    def forward(self, inputs):
        _p = self.classifier(inputs).squeeze(-1).sigmoid()
        return self.dataset.emp_prior * _p / (1 - _p + 1e-6)

    def optimize_callback(self, xk):
        if isinstance(xk, np.ndarray):
            xk = torch.from_numpy(xk)
        xk = xk.reshape(-1, 1)
        xk.add_(0.1 * torch.randn_like(xk))

        yk = torch.zeros(len(xk), 1)

        self.dataset._update_splits(DataSplit(xk, yk))

        ## One stochastic gradient step of the classifier.
        self.classifier.requires_grad_(True)

        num_total = len(self.dataset)
        num_positive = self.dataset.num_positive
        if num_total > 0 and self.dataset.num_positive < num_total:
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [(num_total - num_positive) / num_positive], device=self.device
                )
            )
            loader = torch.utils.data.DataLoader(
                self.dataset, shuffle=True, batch_size=64
            )

            for _ in range(4):
                X, y = next(iter(loader))
                X, y = X.to(self.device).float(), y.to(self.device)
                self.optim.zero_grad()
                loss = loss_fn(self.classifier(X), y)
                loss.backward()
                self.optim.step()

        self.classifier.eval()
        self.classifier.requires_grad_(False)

    def reset_dataset(self):
        self.dataset = RatioEstimator._Dataset()
