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

| Version | Status     | What changed / explored                                                            | Next steps                                |
| ------- | ---------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| v0      | ✅ done    | Baseline: tokenizer + simple bigram model                                          | None. Was just meant for a baseline model |
| v1      | ✅ done    | Shifted to transformer architecture (added multi-head attention, embeddings, etc ) | Add ALiBi and GeLU                        |
| v2      | ✅ done    | Shifted to BPE Tokenization, uses RoPE and GeLU                                    | Experiment with vocab sizes               |
| v3      | 🔄 ongoing | Shifted vocabulary to only English Words, switched from RoPE to AliBi              | Increase Vocab Size + Scaffold-BPE        |
| v4      | 🚧 planned | Undecided                                                                          | Visualize embeddings                      |

## Setup / Requirements

## How to use

## Design & Architecture

## Key Experiments & Results

## Challenges & Learnings

## How to Contribute / Expand

## Liscence

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Scaffold-BPE: Enhancing Byte Pair Encoding for Large Language Models with Simple and Effective Scaffold Token Removal](https://arxiv.org/abs/2404.17808)
- Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT) as inspiration
