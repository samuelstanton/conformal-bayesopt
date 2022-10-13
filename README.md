# [Bayesian Optimization with Conformal Coverage Guarantees](https://arxiv.org/abs/2203.12742)
(fix ArXiv link)

## Abstract

Bayesian optimization is a coherent, ubiquitous approach to decision-making under uncertainty, with applications including multi-arm bandits, active learning, and black-box optimization.
Bayesian optimization selects decisions (i.e. objective function queries) with maximal expected utility with respect to the posterior distribution of a Bayesian model, which quantifies reducible, epistemic uncertainty about query outcomes.
In practice, subjectively implausible outcomes can occur regularly for two reasons: 1) model misspecification and 2) covariate shift.
Conformal prediction is an uncertainty quantification method with coverage guarantees even for misspecified models and a simple mechanism to correct for covariate shift.
We propose conformal Bayesian optimization, which directs queries towards regions of search space where the model predictions have guaranteed validity, and investigate its behavior on a suite of black-box optimization tasks and tabular ranking tasks.
In many cases we find that query coverage can be significantly improved without harming sample-efficiency.

## Main Idea

![Figure 1](https://github.com/samuelstanton/conformal-bayesopt/blob/refactor/conformalbo/assets/figures/branin_example_v0.0.2.png?raw=true)

We want $\mathbf x^* \in [0, 1]^2$ which maximizes the Branin objective **(a)**, starting from $8$ examples in the upper right (the black dots).
The upper-confidence bound (UCB) acquisition function **(b)** selects the next query (the \textcolor{red}{red} star) far from any training data, where we cannot guarantee reliable predictions.
In higher dimensions, we will exhaust our query budget long before covering the whole search space with training data.
Given a miscoverage tolerance $\alpha = 1 / \sqrt{8}$, conformal UCB **(c)** directs the search to the region where conformal predictions are guaranteed coverage of at least $(1 - \alpha)$.
**(d)** The dashed line is the set $\mathbf x$ such that $w(\mathbf x) \propto p_{\mathrm{query}}(\mathbf x) / p_{\mathrm{train}}(\mathbf x)$ is exactly $\alpha$.

## Installation

```bash
git clone https://github.com/samuelstanton/conformal-bayesopt && cd conformal-bayesopt
conda create --name conf-bo-env python=3.8 -y && conda activate conf-bo-env
conda install -c conda-forge rdkit -y
conda install -c pytorch cudatoolkit=11.3
pip install -r requirements.txt --upgrade
pip install -e .
```

## Reproducing the figures

This project uses [Weight and Biases](https://docs.wandb.ai/) for logging.

The experimental data used to produce the plots in our papers is available [here](https://wandb.ai/samuelstanton/conformal-bayesopt).


## Running the code

#### Single-objective, continuous
```bash
python scripts/black_box_opt.py task=ackley acq_fn=cucb
```

#### Multi-objective, continuous
```bash
python scripts/black_box_opt.py task=branin_currin acq_fn=cehvi
```

#### Single-objective, tabular
```bash 
python scripts/tab_bandits.py task=poas_stability acq_fn=cucb
```

## Configuration options

See the config files in `./hydra_config` for all configurable parameters.
Note that any config field can be overridden from the command line, and some configurations are not supported. 

### Task options

#### Single-objective, continuous
- `ackley`
- `branin`
- `levy`
- `michal`

#### Multi-objective, continuous
- `branin_currin`
- `carside`
- `peniciliin`
- `zdt2`

#### Single-objective, tabular
- `poas_hydrophobicity`
- `poas_stability`
- `zinc_penalized_logp`
- `zinc_qed`
- `zinc_3pbl_docking`

### Acquisition options

#### Single-objective
- `cei`
- `cnei`
- `cucb`
- `ei`
- `nei`
- `ucb`

### Multi-objective
- `cehvi`
- `cnehvi`
- `ehvi`
- `nehvi`


## Tests

`pytest tests`

This project currently has very limited test coverage.

## Citation

If you use any part of this code for your own work, please cite
(update citation once ArXiv link is available)

```
@misc{stanton2022bayesian,
      title={Bayesian Optimization with Distribution-Free Coverage Guarantees}, 
      author={Samuel Stanton and Wesley Maddox and Sanyam Kapoor and Andrew Gordon Wilson},
      year={2022},
      eprint={2203.12742},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
