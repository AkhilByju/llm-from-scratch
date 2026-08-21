import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from wikitext.bpe_tokenizer import BPETokenizer


PUNCTUATION_SPACING_PATTERNS = {
    "space_before_period": r"\s+\.",
    "space_before_comma": r"\s+,",
    "space_before_semicolon": r"\s+;",
    "space_before_colon": r"\s+:",
    "space_before_question": r"\s+\?",
    "space_before_exclamation": r"\s+!",
    "space_after_open_paren": r"\(\s+",
    "space_before_close_paren": r"\s+\)",
    "artifact_at_period_at": r"@\.@", 
}


def load_text(path, max_chars=None):
    text = Path(path).read_text(encoding="utf-8")
    if max_chars is not None:
        return text[:max_chars]
    return text


def ngrams(items, n):
    return [tuple(items[i:i + n]) for i in range(max(0, len(items) - n + 1))]


def distinct_n(tokens, n):
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repeated_ngram_rate(tokens, n):
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(grams)


def punctuation_spacing_counts(text):
    return {
        name: len(re.findall(pattern, text))
        for name, pattern in PUNCTUATION_SPACING_PATTERNS.items()
    }


def text_metrics(text):
    words = re.findall(r"\S+", text)
    return {
        "chars": len(text),
        "words": len(words),
        "distinct_1": distinct_n(words, 1),
        "distinct_2": distinct_n(words, 2),
        "repeated_bigram_rate": repeated_ngram_rate(words, 2),
        "repeated_trigram_rate": repeated_ngram_rate(words, 3),
        "punctuation_spacing": punctuation_spacing_counts(text),
    }


def tokenizer_metrics(vocab_path, text):
    tokenizer = BPETokenizer()
    tokenizer.load(vocab_path)
    ids = tokenizer.encode(text, show_progress=False)
    decoded_tokens = [tokenizer.inv_vocab.get(i, "<unk>") for i in ids]
    words = re.findall(r"\S+", text)
    unk_count = sum(1 for token in decoded_tokens if token == "<unk>")
    token_count = len(ids)

    return {
        "vocab_path": str(vocab_path),
        "configured_vocab_size": tokenizer.vocab_size,
        "actual_vocab_entries": len(tokenizer.vocab),
        "sample_chars": len(text),
        "sample_words": len(words),
        "sample_tokens": token_count,
        "tokens_per_word": token_count / max(1, len(words)),
        "unknown_tokens": unk_count,
        "unknown_token_rate": unk_count / max(1, token_count),
        "decoded_text_metrics": text_metrics(tokenizer.decode(ids)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="CPU-cheap diagnostics for nano-scale language model experiments."
    )
    parser.add_argument(
        "--input",
        default="input.txt",
        help="Text file used for tokenizer diagnostics.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50000,
        help="Maximum input characters to evaluate for cheap local runs.",
    )
    parser.add_argument(
        "--vocab",
        action="append",
        default=[],
        help="BPE vocab JSON path. May be repeated.",
    )
    parser.add_argument(
        "--generated-text",
        help="Optional generated text file to score with diversity and punctuation metrics.",
    )
    parser.add_argument(
        "--output",
        default="research/results/metrics.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    text = load_text(args.input, args.max_chars)
    vocab_paths = args.vocab or [
        "Vocabs/bpe_vocab.json",
        "Vocabs/bpe_vocab_500.json",
        "Vocabs/bpe_vocab_english_500.json",
    ]

    result = {
        "input": args.input,
        "max_chars": args.max_chars,
        "input_text_metrics": text_metrics(text),
        "tokenizers": [
            tokenizer_metrics(Path(vocab_path), text)
            for vocab_path in vocab_paths
        ],
    }

    if args.generated_text:
        generated = load_text(args.generated_text)
        result["generated_text"] = {
            "path": args.generated_text,
            "metrics": text_metrics(generated),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
