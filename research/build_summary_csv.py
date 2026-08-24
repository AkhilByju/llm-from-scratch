import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = [
    "generation_seconds",
    "generated_tokens",
    "generated_unknown_tokens",
    "generated_unknown_token_rate",
    "continuation_words",
    "words_per_second",
    "generated_words",
    "generated_bytes",
    "distinct_1",
    "distinct_2",
    "repeated_bigram_rate",
    "repeated_trigram_rate",
    "unknown_markers",
    "unknown_markers_per_100_words",
    "unknown_markers_per_1000_chars",
    "non_ascii_chars",
    "non_ascii_rate",
    "replacement_characters",
    "replacement_character_rate",
    "outside_corpus_chars",
    "outside_corpus_char_rate",
    "space_before_period",
    "space_before_comma",
    "space_before_semicolon",
    "space_before_colon",
    "space_before_question",
    "space_before_exclamation",
    "space_after_open_paren",
    "space_before_close_paren",
    "artifact_at_period_at",
    "total_punctuation_artifacts",
]


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def stddev(rows, field):
    if len(rows) < 2:
        return 0.0
    mu = mean(rows, field)
    variance = sum((float(row[field]) - mu) ** 2 for row in rows) / (len(rows) - 1)
    return math.sqrt(variance)


def total(rows, field):
    return sum(float(row[field]) for row in rows)


def rate_per_100_words(rows, field):
    return total(rows, field) / max(1.0, total(rows, "generated_words")) * 100


def rate_per_1000_chars(rows, field):
    return total(rows, field) / max(1.0, total(rows, "generated_chars")) * 1000


def main():
    parser = argparse.ArgumentParser(
        description="Build model-level summary CSV from comparison.csv."
    )
    parser.add_argument(
        "--input",
        default="research/results/comparison.csv",
        help="Per-prompt comparison CSV path.",
    )
    parser.add_argument(
        "--output",
        default="research/results/comparison_summary.csv",
        help="Output model-level summary CSV path.",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.input).open(encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    summary_rows = []
    for model, model_rows in grouped.items():
        first = model_rows[0]
        summary = {
            "model": model,
            "rows": len(model_rows),
            "params": first["params"],
            "checkpoint_iter": first["checkpoint_iter"],
            "tokenizer": first["tokenizer"],
            "architecture": first["architecture"],
            "positional": first["positional"],
            "context": first["context"],
            "n_embd": first["n_embd"],
            "n_head": first["n_head"],
            "n_layer": first["n_layer"],
            "dropout": first["dropout"],
        }
        for field in NUMERIC_FIELDS:
            summary[f"mean_{field}"] = mean(model_rows, field)
            summary[f"std_{field}"] = stddev(model_rows, field)
        for field in [
            "space_before_period",
            "space_before_comma",
            "artifact_at_period_at",
            "total_punctuation_artifacts",
        ]:
            summary[f"total_{field}"] = total(model_rows, field)
            summary[f"{field}_per_100_words"] = rate_per_100_words(model_rows, field)
            summary[f"{field}_per_1000_chars"] = rate_per_1000_chars(model_rows, field)
        summary_rows.append(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {output_path} with {len(summary_rows)} rows")


if __name__ == "__main__":
    main()
