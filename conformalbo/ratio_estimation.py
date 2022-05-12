import numpy as np
import torch
import math

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor, nn

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
try:
    from optim import SGLD
except ImportError:
    from conformalbo.optim import SGLD
from lambo.utils import DataSplit, update_splits, safe_np_cat


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
    warmup_steps: int = 20,
    sgld_steps: int = 100,
    lr: float = 1e-3,
    temperature: float = 1e-1,
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
    batched_ics = batch_initial_conditions.detach().clone().split(batch_limit)

    # rescale temp so noise has consistent norm
    scaled_temp = temperature / math.sqrt(bounds.size(-1))
    sgld_optimizers = [
        SGLD([ic_batch], lr=lr, momentum=.1, temperature=scaled_temp) for ic_batch in batched_ics
    ]
    # collect SGLD iterates, bootstrap ratio estimator
    sgld_iterates = [
        [] for _ in batched_ics
    ]
    for _s in range(sgld_steps):
        for batch_idx, (ic_batch, batch_optimizer) in enumerate(zip(batched_ics, sgld_optimizers)):
            ic_batch.requires_grad_(True)
            batched_acq_values_ = acq_function(ic_batch)

            if _s >= warmup_steps:
                sgld_iterates[batch_idx].append(ic_batch.detach().clone())
                if callable(options.get('callback')):
                    options.get('callback')(ic_batch.detach().clone().cpu())

            if _s < (sgld_steps - 1):
                batch_optimizer.zero_grad()
                neg_log_density = -batched_acq_values_.sum()
                lb_violation = torch.max(
                    2 * (torch.sigmoid((bounds[0] - ic_batch) / 1.) - 0.5),
                    torch.zeros_like(ic_batch),
                ).sum()
                ub_violation = torch.max(
                    2 * (torch.sigmoid((ic_batch - bounds[1]) / 1.) - 0.5),
                    torch.zeros_like(ic_batch),
                ).sum()
                loss = neg_log_density + 1e2 * (lb_violation + ub_violation)
                loss.backward()
                batch_optimizer.step()
    batched_iterates = [torch.stack(iterates) for iterates in sgld_iterates]
    batch_candidates = torch.cat(batched_iterates, dim=1)

    # final evaluation
    with torch.no_grad():
        batch_shape = batch_candidates.shape[:-2]
        flat_cands = batch_candidates.flatten(0, -3)
        batched_acq_vals = [
            acq_function(cand_batch) for cand_batch in flat_cands.split(batch_limit)
        ]
        flat_acq_vals = torch.cat(batched_acq_vals)
        batch_acq_values = flat_acq_vals.view(*batch_shape)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        # in bounds elementwise?
        in_bounds = (flat_cands >= bounds[0]).prod(-1) * (flat_cands <= bounds[1]).prod(-1)
        # in bounds batchwise?
        in_bounds = in_bounds.prod(-1).bool()

        flat_cands = flat_cands[in_bounds]
        flat_acq_vals = flat_acq_vals[in_bounds]

        best = torch.argmax(flat_acq_vals, dim=0)
        batch_candidates = flat_cands[best]
        batch_acq_values = flat_acq_vals[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values


def optimize_acqf_sgld_list(
    acq_function_list: List[AcquisitionFunction],
    bounds: Tensor,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a list of candidates from a list of acquisition functions.
    The acquisition functions are optimized in sequence, with previous candidates
    set as `X_pending`. This is also known as sequential greedy optimization.
    Args:
        acq_function_list: A list of acquisition functions.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
    Returns:
        A two-element tuple containing
        - a `q x d`-dim tensor of generated candidates.
        - a `q`-dim tensor of expected acquisition values, where the value at
            index `i` is the acquisition value conditional on having observed
            all candidates except candidate `i`.
    """
    if not acq_function_list:
        raise ValueError("acq_function_list must be non-empty.")
    candidate_list, acq_value_list = [], []
    candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)
    base_X_pending = acq_function_list[0].X_pending
    for acq_function in acq_function_list:
        if candidate_list:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
        options = {} if options is None else options
        if hasattr(acq_function.ratio_estimator, 'optimize_callback'):
            options['callback'] = acq_function.ratio_estimator.optimize_callback

        candidate, acq_value = optimize_acqf_sgld(
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            return_best_only=True,
            sequential=False,
        )
        candidate_list.append(candidate)
        acq_value_list.append(acq_value)
        candidates = torch.cat(candidate_list, dim=-2)
    return candidates, torch.stack(acq_value_list)


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
                self._prior_ratio = (n - n_p) / n_p
            return self._prior_ratio

        def _update_splits(self, new_split):
            self.cls_train_split = DataSplit(
                safe_np_cat([self.cls_train_split[0], new_split[0]]),
                safe_np_cat([self.cls_train_split[1], new_split[1]]),
            )
            # self.cls_train_split, self.cls_val_split, self.cls_test_split = update_splits(
            #     train_split=self.cls_train_split,
            #     val_split=self.cls_val_split,
            #     test_split=self.cls_test_split,
            #     new_split=new_split,
            #     holdout_ratio=0.2,
            # )
            self.recompute_emp_prior()

    def __init__(self, in_size=1, device=None, dtype=None):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.in_size = in_size
        self.dataset = RatioEstimator._Dataset()

        ## Remains uniform, when untrained.
        self.classifier = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device=device, dtype=dtype)
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
        # return self.dataset.emp_prior * _p / (1 - _p + 1e-8)
        return _p.clamp_max(1 - 1e-6) / (1 - _p).clamp_min(1e-6)

    def optimize_callback(self, xk):
        if isinstance(xk, np.ndarray):
            xk = torch.from_numpy(xk)
        xk = xk.reshape(-1, self.in_size)
        # xk.add_(0.1 * torch.randn_like(xk))

        yk = torch.ones(xk.size(0), 1)

        self.dataset._update_splits(DataSplit(xk, yk))

        ## One stochastic gradient step of the classifier.
        self.classifier.requires_grad_(True)

        num_total = len(self.dataset)
        num_positive = self.dataset.num_positive
        if num_positive > 0:
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    (self.dataset.emp_prior,), device=self.device
                )
            )
            loader = torch.utils.data.DataLoader(
                self.dataset, shuffle=True, batch_size=num_total
            )

            for _ in range(2):
                X, y = next(iter(loader))
                X = X.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                self.optim.zero_grad()
                loss = loss_fn(self.classifier(X), y)
                loss.backward()
                self.optim.step()

        self.classifier.eval()
        self.classifier.requires_grad_(False)

    def reset_dataset(self):
        self.dataset = RatioEstimator._Dataset()
