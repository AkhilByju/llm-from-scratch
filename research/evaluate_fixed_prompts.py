import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from metrics import text_metrics


DEFAULT_PROMPTS = [
    "The history of language",
    "In the early twentieth century",
    "The city was known for",
    "According to the study",
    "",
]


def run_prompt(command, prompt, timeout):
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        input=f"{prompt}\n",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    stdout = completed.stdout
    stderr = completed.stderr

    marker = "Enter a starting phrase:"
    if marker in stdout:
        generated = stdout.split(marker, 1)[1].strip()
    else:
        generated = stdout.strip()

    if prompt and generated.startswith(prompt):
        continuation = generated[len(prompt):].strip()
    else:
        continuation = generated

    words = continuation.split()
    return {
        "prompt": prompt,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "generated": generated,
        "continuation": continuation,
        "continuation_words": len(words),
        "words_per_second": len(words) / elapsed if elapsed > 0 else math.nan,
        "stdout": stdout,
        "stderr": stderr,
        "metrics": text_metrics(generated),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-prompt generation and cheap diagnostics for saved checkpoints."
    )
    parser.add_argument(
        "--model",
        default="v3_1",
        choices=["v3_1"],
        help="Model runner to evaluate.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the model script.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per prompt in seconds.",
    )
    parser.add_argument(
        "--output",
        default="research/results/fixed_prompt_eval_v3_1.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    if args.model == "v3_1":
        command = [args.python, str(repo_dir / "v3" / "v3_1" / "v3_1.py")]
        checkpoint = "v3/v3_1/best_modelv3_1.pth"
    else:
        raise ValueError(args.model)

    results = []
    for prompt in DEFAULT_PROMPTS:
        print(f"Running prompt: {prompt!r}")
        results.append(run_prompt(command, prompt, args.timeout))

    output = {
        "model": args.model,
        "checkpoint": checkpoint,
        "command": command,
        "prompts": DEFAULT_PROMPTS,
        "results": results,
    }

    output_path = repo_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
