# [Bayesian Optimization with Distribution-Free Coverage Guarantees](https://arxiv.org/abs/2203.12742)
(fix ArXiv link)

## Abstract

Bayesian optimization is a coherent, ubiquitous approach to decision-making under uncertainty, with applications including multi-arm bandits, active learning, and black-box optimization.
Bayesian optimization selects decisions (i.e. queries) with maximal expected utility with respect to the posterior distribution of a Bayesian model which quantifies reducible, epistemic uncertainty about query outcomes.
In practice outcomes with negligible posterior likelihood can occur regularly for two reasons: 1) lack of training data near the Bayes-optimal query and 2) model misspecification.
Conformal prediction is a frequentist procedure that uses imperfect models to make predictions with distribution-free coverage guarantees, with rigorous means of identifying which outcomes cannot be predicted accurately.
In this work we propose conformal Bayesian optimization, which directs queries towards regions of search space where the model is valid, and demonstrates significantly improved query coverage and competitive sample-efficiency on a suite of single and multi-objective tasks.

## Key Results

replace figure

![Figure 1](https://github.com/samuelstanton/lambo/blob/main/lambo/assets/figures/lambo_pareto_front.png?raw=true)


## Installation

```bash
git clone https://github.com/samuelstanton/conformal-bayesopt && cd conformal-bayesopt
conda create --name conf-bo-env python=3.8 -y && conda activate conf-bo-env
conda install -c conda-forge rdkit -y
conda install -c pytorch torchvision cudatoolkit=11.3
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
