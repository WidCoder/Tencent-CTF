#!/usr/bin/env python3
import argparse
import glob
import json
from collections import defaultdict
from statistics import mean, median

from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"


def collect_text(value):
    texts = []
    if isinstance(value, str):
        texts.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            texts.extend(collect_text(item))
    elif isinstance(value, list):
        for item in value:
            texts.extend(collect_text(item))
    return texts


def load_tokenizer(name_or_path):
    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)


def iter_input_paths(patterns):
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            for path in matches:
                if path not in seen:
                    seen.add(path)
                    yield path
        elif pattern not in seen:
            seen.add(pattern)
            yield pattern


def count_record_tokens(tokenizer, raw_line, mode):
    if mode == "raw_json":
        return len(tokenizer.encode(raw_line, add_special_tokens=False)), [], defaultdict(lambda: {"messages": 0, "tokens": 0}), {}

    record = json.loads(raw_line)
    message_tokens = []
    role_stats = defaultdict(lambda: {"messages": 0, "tokens": 0})

    for message in record.get("messages", []):
        role = message.get("role", "unknown")
        payload = {k: v for k, v in message.items() if k != "role"}
        texts = collect_text(payload)
        joined = "\n".join(texts)
        token_count = len(tokenizer.encode(joined, add_special_tokens=False))
        message_tokens.append(token_count)
        role_stats[role]["messages"] += 1
        role_stats[role]["tokens"] += token_count

    total_tokens = sum(message_tokens)
    return total_tokens, message_tokens, role_stats, record


def summarize_message_tokens(message_tokens):
    if not message_tokens:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {
        "mean": round(mean(message_tokens), 2),
        "median": round(median(message_tokens), 2),
        "min": min(message_tokens),
        "max": max(message_tokens),
    }


def main():
    parser = argparse.ArgumentParser(description="Count tokens in trajectory JSONL files with a Qwen2.5 tokenizer.")
    parser.add_argument("paths", nargs="+", help="Input JSONL file paths or glob patterns.")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer name or local path (default: {DEFAULT_TOKENIZER}).",
    )
    parser.add_argument(
        "--mode",
        choices=["texts", "raw_json"],
        default="texts",
        help="Count text fields only or count the raw JSON line.",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)

    record_index = 0
    grand_total = 0
    grand_messages = 0
    grand_message_tokens = []

    for path in iter_input_paths(args.paths):
        with open(path, "r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                raw_line = raw_line.rstrip("\n\r")
                if not raw_line.strip():
                    continue

                record_index += 1
                total_tokens, message_tokens, role_stats, record = count_record_tokens(tokenizer, raw_line, args.mode)

                grand_total += total_tokens
                grand_messages += len(message_tokens)
                grand_message_tokens.extend(message_tokens)

                record_id = record.get("id", f"record-{record_index}")
                sample_type = record.get("sample_type", "")
                stats = summarize_message_tokens(message_tokens)

                print(
                    f"[record {record_index}] id={record_id} sample_type={sample_type} "
                    f"tokens={total_tokens} messages={len(message_tokens)} "
                    f"avg_msg={stats['mean']} median_msg={stats['median']} "
                    f"min_msg={stats['min']} max_msg={stats['max']}"
                )
                if args.mode == "texts":
                    for role in sorted(role_stats):
                        rs = role_stats[role]
                        avg_role = round(rs["tokens"] / rs["messages"], 2) if rs["messages"] else 0.0
                        print(f"  role={role} messages={rs['messages']} tokens={rs['tokens']} avg={avg_role}")

    if record_index:
        overall = summarize_message_tokens(grand_message_tokens)
        avg_per_record = round(grand_total / record_index, 2)
        avg_per_message = round(grand_total / grand_messages, 2) if grand_messages else 0.0
        print("[overall]")
        print(
            f"records={record_index} messages={grand_messages} tokens={grand_total} "
            f"avg_tokens_per_record={avg_per_record} avg_tokens_per_message={avg_per_message} "
            f"message_mean={overall['mean']} message_median={overall['median']} "
            f"message_min={overall['min']} message_max={overall['max']}"
        )


if __name__ == "__main__":
    main()
