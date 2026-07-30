# Reinforcement Learning in Portfolio Management

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21541556-1682D4?style=flat-square)](https://doi.org/10.5281/zenodo.21541556)
[![PyTorch](https://img.shields.io/badge/PyTorch-preview-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Pylint](https://github.com/chriss1245/reinforcement_learning_in_finance/actions/workflows/norm_checking.yml/badge.svg)](https://github.com/chriss1245/reinforcement_learning_in_finance/actions/workflows/norm_checking.yml)
[![Unit Tests](https://github.com/chriss1245/reinforcement_learning_in_finance/actions/workflows/unittests.yml/badge.svg)](https://github.com/chriss1245/reinforcement_learning_in_finance/actions/workflows/unittests.yml)

Deep Reinforcement Learning agents that manage a stock portfolio end to end — deciding
not just *how* to rebalance, but *whether* to rebalance at all.

This is the codebase behind my bachelor thesis at **Universidad Carlos III de Madrid**
(Bachelor in Data Science and Engineering, defended September 2023), supervised by
Francisco Javier Nogales Martín.

📄 **[Read the thesis](https://doi.org/10.5281/zenodo.21541556)** — archived on Zenodo with a DOI
&nbsp;|&nbsp; 🌐 **[Overview on manapple.dev](https://manapple.dev/academia)**
&nbsp;|&nbsp; [en español](https://manapple.dev/es/academia)

---

## The problem

Most machine learning in finance ends up as a supplementary tool for a human expert: it
forecasts a return, scores a signal, flags a risk — and a person still makes the call.
This work asks what it takes to close that loop and let the model own the allocation
decision itself.

## Approach

| Component | Choice |
|---|---|
| State encoder | Temporal Convolutional Network, pre-trained on supervised return forecasting, then **frozen** and transferred — representation learning kept separate from policy learning |
| Agent | Soft Actor-Critic (**SACv2**) |
| Actor | Normal and Dirichlet distributions over the portfolio simplex, so allocations are valid weights by construction |
| Critic | Bilinear and single-head attention variants |
| Rebalance decision | A **confidence score** lets the agent skip rebalancing, making it agnostic to the decision interval rather than tied to a fixed cadence |
| Regularisation | Random asset permutation (company-agnostic policies) + mixup across price channels |

## Evaluation

S&P 500 data, split chronologically so no future information leaks backwards:

| Split | Period |
|---|---|
| Training | 1986 – 2008 |
| Validation | 2009 – 2016 |
| Test | 2017 – 2023, **unseen companies** |

Each partition contains periods of market turbulence. Agents are benchmarked against
equal-weight (with and without rebalancing), minimum-volatility and maximum-Sharpe
efficient-frontier portfolios, and a random agent.

**Stated limitations:** the environment is simulated, and the dataset carries
survivorship bias. Both are discussed in the thesis rather than papered over.

## What's in the repo

- Stock data collection and processing pipelines
- The DRL agents, plus the benchmark agents they are measured against
- A unified backtesting harness so every agent is evaluated identically
- CI for unit tests and lint (Pylint, Black)

## Citing this work

> Manzano Vimos, C. A. (2023). *Reinforcement Learning in Portfolio Management*
> [Bachelor Thesis]. Universidad Carlos III de Madrid. Zenodo.
> https://doi.org/10.5281/zenodo.21541556

Licensed CC BY-NC-ND 4.0.

---

Built by **[Christopher Manzano Vimos](https://manapple.dev)** — Data Scientist & AI Engineer.
Available for [talks, mentoring and consulting](https://manapple.dev/services).
