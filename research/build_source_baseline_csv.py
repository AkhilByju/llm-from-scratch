import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Flatten source-corpus baseline metrics to CSV.")
    parser.add_argument("--input", default="research/results/metrics.json")
    parser.add_argument("--output", default="research/results/source_baseline.csv")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    metrics = data["input_text_metrics"]
    row = {
        "input": data["input"],
        "max_chars": data["max_chars"],
        "start_byte": data.get("start_byte"),
        "max_bytes": data.get("max_bytes"),
        "source_start_byte": data.get("source_start_byte"),
        "source_end_byte": data.get("source_end_byte"),
        "source_corpus_character_count": data["source_corpus_character_count"],
        "chars": metrics["chars"],
        "bytes": metrics["bytes"],
        "words": metrics["words"],
        "distinct_1": metrics["distinct_1"],
        "distinct_2": metrics["distinct_2"],
        "repeated_bigram_rate": metrics["repeated_bigram_rate"],
        "repeated_trigram_rate": metrics["repeated_trigram_rate"],
    }
    for field, count in metrics["punctuation_spacing"].items():
        row[field] = count
        row[f"{field}_per_100_words"] = count / max(1, metrics["words"]) * 100
        row[f"{field}_per_1000_chars"] = count / max(1, metrics["chars"]) * 1000
    for field, value in metrics["unknown_markers"].items():
        row[field] = value
    for field, value in metrics["character_fragmentation"].items():
        row[field] = value

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
