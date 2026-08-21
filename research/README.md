# Research

This folder turns the exploratory WikiText experiments into a reproducible study of nano-scale language models trained from scratch.

The current paper direction is:

> How do tokenization choice, vocabulary restriction, context length, and positional encoding affect nano-scale transformer language models trained from scratch on WikiText-like data?

## Files

- `STUDY_PLAN.md`: research question, claims, metric protocol, and current findings.
- `metrics.py`: cheap tokenizer and text diagnostics.
- `evaluate_fixed_prompts.py`: fixed-prompt generation evaluation for saved checkpoints.
- `results/`: generated JSON outputs from the evaluation scripts.

## Run

Use the Anaconda Python because the local `.venv` does not currently have PyTorch installed.

```bash
/opt/anaconda3/bin/python research/metrics.py --max-chars 50000
/opt/anaconda3/bin/python research/evaluate_fixed_prompts.py --python /opt/anaconda3/bin/python
```

## Publication Boundary

This folder is suitable for the public GitHub repository. It contains experiment code, aggregate metrics, and generated samples. Keep private material elsewhere if it includes reviewer responses, private venue correspondence, unfinished anonymous submissions, or notes you do not want publicly associated with the project.
