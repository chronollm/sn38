# SN38 — ChronoLLM

**Bittensor Subnet 38** — Competitive training of chronologically consistent Large Language Models.

> Currently in testing phase on mainnet.

[![Discord](https://img.shields.io/discord/799672011265015819?label=Discord&logo=discord)](https://discord.com/channels/799672011265015819/1485634202895519844)

## The problem

Standard LLMs are trained on data from all time periods. When used for financial backtesting or historical analysis, they suffer from **lookahead bias** — the model "knows" things that weren't known at the time being analyzed. A model asked to predict 2015 market trends has already seen the 2020 crash.

## The solution

Miners train a collection of language models, one per year (2013-2025). Each model must be trained **only on data available up to its cutoff year**. A model for 2018 must know nothing about 2019 or beyond.

## How it works

```
Miners                          Validators (TEE)                    Backend (private)
  |                                   |                                    |
  |  Train models per year            |                                    |
  |  Upload to HuggingFace            |                                    |
  |  Submit on-chain                  |                                    |
  |  ------------------------------>  |                                    |
  |                                   |  Download models                   |
  |                                   |  Fetch private evaluation data     |
  |                                   |  <---------------------------------|
  |                                   |                                    |
  |                                   |  Stage 1: Check chronological      |
  |                                   |           consistency per year     |
  |                                   |                                    |
  |                                   |  Stage 2: Quality evaluation       |
  |                                   |           via LLM judge duels      |
  |                                   |                                    |
  |                                   |  Set weights on-chain              |
  |                                   |                                    |
```

## Scoring

Evaluation happens in two stages:

**Stage 1 — Consistency check (all miners)**

Each model is tested against a private evaluation dataset to verify it doesn't contain knowledge from after its cutoff year. The evaluation dataset is kept secret to prevent gaming — validators run inside a Trusted Execution Environment (TEE) so that neither the validator operator nor the miners can access the dataset.

Each year produces a score. Missing years, oversized models, or errors receive the worst possible score. The top 10 miners qualify for Stage 2.

Downloading, loading, and evaluating miners' models is parallelized across and within miners — see [docs/validator.md](docs/validator.md#stage-1-parallelization) for details and tuning knobs.

**Stage 2 — Quality evaluation (top 10)**

Qualified miners' models answer open-ended questions. An LLM judge compares every pair of miners in a round-robin tournament. The win rate becomes the quality score.

**Final score**

```
final = 0.7 * leak_score + 0.3 * quality_win_rate
```

Winner takes all — only the highest final score earns emissions.

## Rounds

Rounds last **one week**, starting every Monday at 12:00 UTC.

```
Monday 12:00 UTC                    Monday 12:00 UTC                    Monday 12:00 UTC
       |-------- Round N ------------------|-------- Round N+1 ------------------|
       |                                   |                                     |
       |  Miners: submit for Round N       |  Miners: submit for Round N+1       |
       |  Validators: evaluate Round N-1   |  Validators: evaluate Round N       |
```

During each round, miners can submit or update their models. At the same time, validators evaluate the submissions from the **previous round**. This overlap means miners always have a full week to prepare while validators work on a stable snapshot.

## Why TEE

The evaluation dataset must stay private. If miners knew exactly what questions are used, they could overfit their models to pass the tests without truly maintaining chronological consistency.

Validators run inside [Phala Cloud](https://phala.network/) using Intel TDX hardware. The TEE guarantees that:

1. The evaluation data is only accessible inside the secure enclave
2. The validator code is auditable via remote attestation
3. Neither the validator operator nor any miner can read the evaluation data

Anyone can verify the attestation to confirm the validator is running the correct, unmodified code.

## Model requirements

- Must use the ChronoGPT architecture (a chain of yearly model implementing the [ChronoGPT](sn38/template/chronogpt_model.py) class)
- Safetensors format (no pickle)
- Maximum 2B parameters - initial goal is too reduce the time leaks and improve the quality of the model while staying small and efficient.

## Getting started

| | |
|--|--|
| **Mine** | [docs/miner.md](docs/miner.md) |
| **Validate** | [docs/validator.md](docs/validator.md) |
| ChronoGPT paper | [arXiv:2510.11677](https://arxiv.org/abs/2510.11677) |
| ChronoGPT models | [huggingface.co/manelalab](https://huggingface.co/manelalab) |
| Bittensor docs | [docs.learnbittensor.org](https://docs.learnbittensor.org/) |

## License

This repository is licensed under the MIT License. Full details are available in the [LICENSE](LICENSE) file.
