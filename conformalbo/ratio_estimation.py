import torch
from torch import nn
import numpy as np

from lambo.utils import DataSplit, update_splits


class RatioEstimator(nn.Module):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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

    def __init__(self, in_size=1):
        super().__init__()

        self.dataset = RatioEstimator._Dataset()

        ## Remains uniform, when untrained.
        self.classifier = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
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
        # xk.add_(0.1 * torch.randn_like(xk))

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
