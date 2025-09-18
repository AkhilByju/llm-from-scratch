# LLM From Scratch

Goal: Implement a transformer-based LLM completely form scratch \
Motivation: learn internals of tokenization, training loops, attention mechanisms, etc. \
Scope: pretraining on WikiText, experimenting with hyperparameters \

## Table of Contents

- [Motivation](#motivation)
- [What’s in this repo](#whats-in-this-repo)
- [Versions / Roadmap](#versions--roadmap)
- [Setup / Requirements](#setup--requirements)
- [How to use](#how-to-use)
- [Design & Architecture](#design--architecture)
- [Key Experiments & Results](#key-experiments--results)
- [Challenges & Learnings](#challenges--learnings)
- [How to Contribute / Expand](#how-to-contribute--expand)
- [License](#license)
- [References](#references)

## Motivation

- I want to learn and explore Large Language Models
- Understand tokenization and vocobulation methods
- Build progressively more capable models to better understand LLMs
- Compare different architectures and positional encodings (e.g., ALiBi vs. RoPE)

## What's in this repo

- `v0.py` — baseline model
- `v1.py`, `v1.1.py`, `v2.py`, `v3.py` — iterative improvements
- `bpe_tokenizer.py` — Byte Pair Encoding implementation
- `bpe_vocab.json` / `bpe_vocab_500.json` — vocab artifacts
- `input.txt` — sample dataset
- `train_ids.pt`, `val_ids.pt` — preprocessed training/validation sets
- Notebooks — prototyping & analysis

## Versions / Roadmap

| Version | Status     | What changed / explored                    | Next steps                  |
| ------- | ---------- | ------------------------------------------ | --------------------------- |
| v0      | ✅ done    | Baseline: tokenizer + simple model         | Scale to larger dataset     |
| v1      | ✅ done    | Improved tokenizer, batching               | Optimize memory usage       |
| v2      | ✅ done    | Added transformer attention                | Experiment with vocab sizes |
| v3      | 🔄 ongoing | Scaling layers, experimenting with dropout | Add evaluation metrics      |
| v4      | 🚧 planned | Integrate ALiBi, compare with RoPE         | Visualize embeddings        |

## Setup / Requirements

## How to use

## Design & Architecture

## Key Experiments & Results

## Challenges & Learnings

## How to Contribute / Expand

## Liscence

## References
