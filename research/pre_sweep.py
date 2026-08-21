import argparse
import json
import time
from pathlib import Path

import torch

from metrics import text_metrics
from model_registry import MODEL_SPECS, REPO_DIR, load_model, write_registry


def safe_encode(tokenizer, prompt):
    ids = tokenizer.encode(prompt)
    if not ids:
        return [0]
    return ids


def run_model(name, prompt, max_new_tokens, device):
    started = time.perf_counter()
    result = {
        "model": name,
        "ok": False,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    }
    try:
        model, tokenizer, metadata = load_model(name, device=device)
        load_seconds = time.perf_counter() - started

        gen_started = time.perf_counter()
        context = torch.tensor([safe_encode(tokenizer, prompt)], dtype=torch.long, device=device)
        generated_ids = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
        generated = tokenizer.decode(generated_ids)
        gen_seconds = time.perf_counter() - gen_started

        continuation = generated[len(prompt):].strip() if prompt and generated.startswith(prompt) else generated
        words = continuation.split()

        result.update(
            {
                "ok": True,
                "checkpoint_iter": metadata["iter"],
                "params": metadata["params"],
                "spec": metadata["spec"],
                "load_seconds": load_seconds,
                "generation_seconds": gen_seconds,
                "continuation_words": len(words),
                "words_per_second": len(words) / gen_seconds if gen_seconds else None,
                "generated": generated,
                "metrics": text_metrics(generated),
            }
        )
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pre-sweep checkpoint load and generation smoke test."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_SPECS.keys()),
        help="Model registry names to test.",
    )
    parser.add_argument(
        "--prompt",
        default="The history of language",
        help="Prompt used for generation smoke tests.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Short generation length for cheap smoke tests.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="research/results/pre_sweep.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    output = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "results": [],
    }

    for name in args.models:
        print(f"Testing {name}...")
        output["results"].append(
            run_model(name, args.prompt, args.max_new_tokens, args.device)
        )

    output_path = REPO_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_registry(REPO_DIR / "research" / "results" / "model_registry.json")

    failures = [r for r in output["results"] if not r["ok"]]
    print(f"Wrote {output_path}")
    print(f"Passed {len(output['results']) - len(failures)}/{len(output['results'])}")
    if failures:
        for failure in failures:
            print(f"FAILED {failure['model']}: {failure['error_type']}: {failure['error']}")


if __name__ == "__main__":
    main()
