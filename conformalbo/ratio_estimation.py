import numpy as np
import torch
import copy

from torch import nn

from conformalbo.utils import DataSplit, update_splits, safe_np_cat


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

    def __init__(self, in_size=1, device=None, dtype=None, ema_weight=1e-2, lr=1e-3, weight_decay=1e-4):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.in_size = in_size
        self.dataset = RatioEstimator._Dataset()
        self._neg_samples = None
        self._pos_samples = []

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_size, 1),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device=device, dtype=dtype)

        ## density ratio estimates are exactly 1 when untrained
        self._target_network = copy.deepcopy(self.classifier)
        self._target_network.requires_grad_(False)
        for tgt_p in self._target_network.parameters():
            tgt_p.data.fill_(0.)
        self._ema_weight = ema_weight

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)

    def set_neg_samples(self, neg_samples):
        self._neg_samples = neg_samples

    def set_pos_samples(self, pos_samples):
        self._pos_samples

    def forward(self, inputs):
        _p = self._target_network(inputs).squeeze(-1).sigmoid()
        return _p.clamp_max(1 - 1e-6) / (1 - _p).clamp_min(1e-6)

    def update_target_network(self):
        with torch.no_grad():
            for src_p, tgt_p in zip(
                    self.classifier.parameters(), self._target_network.parameters()
            ):
                tgt_p.mul_(1. - self._ema_weight)
                tgt_p.add_(self._ema_weight * src_p)


    def optimize_callback(self, xk):
        if isinstance(xk, np.ndarray):
            xk = torch.from_numpy(xk)
        xk = xk.reshape(-1, self.in_size)
        self._pos_samples.extend([x for x in xk])

        if self._neg_samples is None:
            return None

        num_negative = self._neg_samples.size(0)
        num_positive = len(self._pos_samples)

        if num_positive < num_negative:
            return None

        self.classifier.requires_grad_(True)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        neg_minibatch = self._neg_samples.to(device=self.device, dtype=self.dtype)
        pos_minibatch = torch.stack([
            self._pos_samples[idx] for idx in np.random.permutation(num_positive)[:num_negative]
        ]).to(device=self.device, dtype=self.dtype)
        minibatch_X = torch.cat([neg_minibatch, pos_minibatch])
        minibatch_Z = torch.cat(
            [torch.zeros(num_negative, 1), torch.ones(num_negative, 1)]
        ).to(device=self.device, dtype=self.dtype)

        self.optim.zero_grad()
        loss = loss_fn(self.classifier(minibatch_X), minibatch_Z)
        loss.backward()
        self.optim.step()
        self.update_target_network()

        self.classifier.eval()
        self.classifier.requires_grad_(False)

        # xk.add_(0.1 * torch.randn_like(xk))
        # yk = torch.ones(xk.size(0), 1)

        # self.dataset._update_splits(DataSplit(xk, yk))

        ## One stochastic gradient step of the classifier.
        # self.classifier.requires_grad_(True)

        # num_total = len(self.dataset)
        
        # if num_positive > 0:
        #     loss_fn = torch.nn.BCEWithLogitsLoss(
        #         pos_weight=torch.tensor(
        #             (self.dataset.emp_prior,), device=self.device
        #         )
        #     )
        #     loader = torch.utils.data.DataLoader(
        #         self.dataset, shuffle=True, batch_size=num_total
        #     )

        #     for _ in range(1):
        #         X, y = next(iter(loader))
        #         X = X.to(device=self.device, dtype=self.dtype)
        #         y = y.to(device=self.device, dtype=self.dtype)
        #         self.optim.zero_grad()
        #         loss = loss_fn(self.classifier(X), y)
        #         loss.backward()
        #         self.optim.step()
        #         self.update_target_network()

        # self.classifier.eval()
        # self.classifier.requires_grad_(False)

    def reset_dataset(self):
        self.dataset = RatioEstimator._Dataset()
