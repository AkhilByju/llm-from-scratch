import argparse
import csv
import json
from pathlib import Path


PUNCTUATION_FIELDS = [
    "space_before_period",
    "space_before_comma",
    "space_before_semicolon",
    "space_before_colon",
    "space_before_question",
    "space_before_exclamation",
    "space_after_open_paren",
    "space_before_close_paren",
    "artifact_at_period_at",
]


def row_for_result(model_result, prompt_result):
    spec = model_result["spec"]
    metrics = prompt_result["metrics"]
    punctuation = metrics["punctuation_spacing"]
    unknown_markers = metrics["unknown_markers"]
    fragmentation = metrics["character_fragmentation"]

    row = {
        "model": model_result["model"],
        "prompt": prompt_result["prompt"] or "<empty>",
        "seed": prompt_result.get("seed"),
        "ok": model_result["ok"],
        "params": model_result["params"],
        "checkpoint_iter": model_result["checkpoint_iter"],
        "tokenizer": spec["tokenizer"],
        "architecture": spec["architecture"],
        "positional": spec["positional"],
        "context": spec["block_size"],
        "n_embd": spec["n_embd"],
        "n_head": spec["n_head"],
        "n_layer": spec["n_layer"],
        "dropout": spec["dropout"],
        "generation_seconds": prompt_result["elapsed_seconds"],
        "prompt_tokens": prompt_result.get("prompt_tokens"),
        "generated_tokens": prompt_result.get("generated_tokens"),
        "generated_unknown_tokens": prompt_result.get("generated_unknown_tokens", 0),
        "generated_unknown_token_rate": prompt_result.get("generated_unknown_token_rate", 0),
        "continuation_words": prompt_result["continuation_words"],
        "words_per_second": prompt_result["words_per_second"],
        "generated_chars": metrics["chars"],
        "generated_bytes": metrics["bytes"],
        "generated_words": metrics["words"],
        "distinct_1": metrics["distinct_1"],
        "distinct_2": metrics["distinct_2"],
        "repeated_bigram_rate": metrics["repeated_bigram_rate"],
        "repeated_trigram_rate": metrics["repeated_trigram_rate"],
        "unknown_markers": unknown_markers["unknown_markers"],
        "unknown_markers_per_100_words": unknown_markers["unknown_markers_per_100_words"],
        "unknown_markers_per_1000_chars": unknown_markers["unknown_markers_per_1000_chars"],
        "non_ascii_chars": fragmentation["non_ascii_chars"],
        "non_ascii_rate": fragmentation["non_ascii_rate"],
        "replacement_characters": fragmentation["replacement_characters"],
        "replacement_character_rate": fragmentation["replacement_character_rate"],
        "outside_corpus_chars": fragmentation["outside_corpus_chars"],
        "outside_corpus_char_rate": fragmentation["outside_corpus_char_rate"],
    }
    for field in PUNCTUATION_FIELDS:
        row[field] = punctuation.get(field, 0)
        row[f"{field}_per_100_words"] = (
            punctuation.get(field, 0) / max(1, metrics["words"]) * 100
        )
        row[f"{field}_per_1000_chars"] = (
            punctuation.get(field, 0) / max(1, metrics["chars"]) * 1000
        )
    total_punctuation_artifacts = sum(punctuation.get(field, 0) for field in PUNCTUATION_FIELDS)
    row["total_punctuation_artifacts"] = total_punctuation_artifacts
    row["total_punctuation_artifacts_per_100_words"] = (
        total_punctuation_artifacts / max(1, metrics["words"]) * 100
    )
    row["total_punctuation_artifacts_per_1000_chars"] = (
        total_punctuation_artifacts / max(1, metrics["chars"]) * 1000
    )
    row["generated"] = prompt_result["generated"]
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Flatten fixed-prompt evaluation JSON into comparison CSV."
    )
    parser.add_argument(
        "--input",
        default="research/results/fixed_prompt_eval_full.json",
        help="Evaluation JSON path.",
    )
    parser.add_argument(
        "--output",
        default="research/results/comparison.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    rows = []
    for model_result in data["results"]:
        if not model_result["ok"]:
            rows.append(
                {
                    "model": model_result["model"],
                    "ok": False,
                    "error_type": model_result.get("error_type"),
                    "error": model_result.get("error"),
                }
            )
            continue
        for prompt_result in model_result["results"]:
            rows.append(row_for_result(model_result, prompt_result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
