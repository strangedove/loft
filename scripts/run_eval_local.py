#!/usr/bin/env python3
"""Eval a local llama-server model via OpenAI-compatible completions API.

Runs each prompt N times and aggregates results. Supports --thinking flag
to compare reasoning-enabled vs reasoning-disabled outputs.

Usage:
    python run_eval_local.py --runs 3 --label qwen35-27b-step1
    python run_eval_local.py --runs 3 --label qwen35-27b-step1 --thinking
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "loft" / "eval" / "test_prompts.json"

TEMPERATURE = 0.9
MIN_P = 0.03
REPETITION_PENALTY = 1.0


def build_chatml_prompt(messages, thinking=False):
    """Build chatml-formatted prompt.

    thinking=False: ends with empty think block (reasoning disabled)
    thinking=True:  ends with open think tag (reasoning enabled)
    """
    chatml = ""
    for msg in messages:
        chatml += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    if thinking:
        chatml += "<|im_start|>assistant\n<think>\n"
    else:
        chatml += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return chatml


def generate(client, prompt, model_id, max_tokens):
    """Generate a completion using the completions API."""
    resp = client.post("/v1/completions", json={
        "model": model_id,
        "prompt": prompt,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "min_p": MIN_P,
        "repeat_penalty": REPETITION_PENALTY,
        "stop": ["<|im_end|>", "<|im_start|>"],
    })
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    finish = data["choices"][0].get("finish_reason", "unknown")
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return text.strip(), finish, tokens


def main():
    parser = argparse.ArgumentParser(description="Eval local llama-server model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090", help="llama-server URL")
    parser.add_argument("--model", default=None, help="Model ID (auto-detected if omitted)")
    parser.add_argument("--label", default="eval", help="Label for output files")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "evals"), help="Output directory")
    parser.add_argument("--categories", nargs="*", default=None, help="Only run these categories")
    parser.add_argument("--thinking", action="store_true", help="Enable reasoning (open think tag, 3000 max tokens)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max output tokens")
    args = parser.parse_args()

    # Determine max tokens based on thinking mode
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    elif args.thinking:
        max_tokens = 3000
    else:
        max_tokens = 512

    # Auto-suffix the label
    if args.thinking and not args.label.endswith("-think"):
        args.label += "-think"
    elif not args.thinking and not args.label.endswith("-nothink"):
        args.label += "-nothink"

    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect model
    if args.model is None:
        resp = httpx.get(f"{base_url}/v1/models", timeout=30)
        resp.raise_for_status()
        models = resp.json()["data"]
        model_id = models[0]["id"]
        print(f"Model: {model_id}")
    else:
        model_id = args.model

    with open(PROMPTS_PATH) as f:
        all_prompts = json.load(f)
    if args.categories:
        all_prompts = [p for p in all_prompts if p["category"] in args.categories]

    mode_str = "thinking ON" if args.thinking else "thinking OFF"
    print(f"Mode: {mode_str} | max_tokens={max_tokens} | temp={TEMPERATURE} | min_p={MIN_P}")
    print(f"Loaded {len(all_prompts)} prompts, {args.runs} runs each = {len(all_prompts) * args.runs} total generations\n")

    client = httpx.Client(base_url=base_url, timeout=300)

    # Structure: {prompt_id: {category, prompt_messages, runs: [{output, ...}]}}
    all_results = {}

    for run_idx in range(args.runs):
        print(f"{'='*60}")
        print(f"RUN {run_idx + 1}/{args.runs}")
        print(f"{'='*60}")

        for p in all_prompts:
            pid = p["id"]
            cat = p["category"]
            prompt = build_chatml_prompt(p["messages"], thinking=args.thinking)

            print(f"  [{pid}]", end=" ", flush=True)
            t0 = time.time()

            try:
                text, finish, tokens = generate(client, prompt, model_id, max_tokens)
            except Exception as e:
                print(f"ERROR: {e}")
                text = f"[ERROR: {e}]"
                finish = "error"
                tokens = 0

            elapsed = time.time() - t0

            if pid not in all_results:
                all_results[pid] = {
                    "category": cat,
                    "messages": p["messages"],
                    "runs": [],
                }

            all_results[pid]["runs"].append({
                "run": run_idx + 1,
                "output": text,
                "finish_reason": finish,
                "tokens": tokens,
                "elapsed": round(elapsed, 1),
            })

            preview = text[:100].replace("\n", " ")
            print(f"{tokens}tok {elapsed:.1f}s | {preview}...")

        print()

    # Compute aggregated statistics
    print(f"\n{'='*60}")
    print("COMPUTING AGGREGATED RESULTS")
    print(f"{'='*60}\n")

    # Per-prompt aggregation
    prompt_stats = {}
    for pid, data in all_results.items():
        word_counts = [len(r["output"].split()) for r in data["runs"]]
        token_counts = [r["tokens"] for r in data["runs"]]

        prompt_stats[pid] = {
            "category": data["category"],
            "mean_words": round(sum(word_counts) / len(word_counts)),
            "mean_tokens": round(sum(token_counts) / len(token_counts)),
        }

    # Per-category aggregation
    cat_stats = {}
    for pid, stats in prompt_stats.items():
        cat = stats["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"word_counts": [], "prompts": []}
        cat_stats[cat]["word_counts"].append(stats["mean_words"])
        cat_stats[cat]["prompts"].append(pid)

    for cat in cat_stats:
        wc = cat_stats[cat]["word_counts"]
        cat_stats[cat]["mean_words"] = round(sum(wc) / len(wc))

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_base = output_dir / f"eval-{args.label}-{timestamp}"

    json_output = {
        "model": model_id,
        "label": args.label,
        "thinking": args.thinking,
        "runs": args.runs,
        "sampling": {
            "temperature": TEMPERATURE,
            "min_p": MIN_P,
            "repetition_penalty": REPETITION_PENALTY,
            "max_tokens": max_tokens,
        },
        "results": {pid: {
            "category": data["category"],
            "messages": data["messages"],
            "runs": data["runs"],
        } for pid, data in all_results.items()},
        "prompt_stats": prompt_stats,
        "category_stats": {cat: {k: v for k, v in s.items() if k != "prompts"}
                           for cat, s in cat_stats.items()},
    }
    json_path = f"{output_base}.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    # Build markdown report
    md_path = f"{output_base}.md"
    with open(md_path, "w") as f:
        f.write(f"# Eval Report: {args.label}\n\n")
        f.write(f"**Model**: {model_id}\n")
        f.write(f"**Thinking**: {'enabled' if args.thinking else 'disabled'}\n")
        f.write(f"**Runs per prompt**: {args.runs}\n")
        f.write(f"**Sampling**: temp={TEMPERATURE}, min_p={MIN_P}, rep_penalty={REPETITION_PENALTY}, max_tokens={max_tokens}\n\n")

        f.write("## Summary by Category\n\n")
        f.write("| Category | Prompts | Mean words |\n")
        f.write("|----------|--------:|-----------:|\n")
        for cat in sorted(cat_stats.keys()):
            s = cat_stats[cat]
            f.write(f"| {cat} | {len(s['prompts'])} | {s['mean_words']} |\n")
        f.write("\n")

        f.write("## Summary by Prompt\n\n")
        f.write("| Prompt | Category | Mean words | Mean tokens |\n")
        f.write("|--------|----------|--------:|-----------:|\n")
        for p in all_prompts:
            pid_id = p["id"]
            if pid_id in prompt_stats:
                ps = prompt_stats[pid_id]
                f.write(f"| {pid_id} | {ps['category']} | {ps['mean_words']} | {ps['mean_tokens']} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## Full Outputs\n\n")

        for p in all_prompts:
            pid = p["id"]
            if pid not in all_results:
                continue
            data = all_results[pid]
            cat = data["category"]
            user_msgs = [m for m in p["messages"] if m["role"] == "user"]
            sys_msgs = [m for m in p["messages"] if m["role"] == "system"]
            user_msg = user_msgs[-1]["content"] if user_msgs else ""
            sys_msg = sys_msgs[0]["content"] if sys_msgs else ""

            f.write(f"### {pid} ({cat})\n\n")
            if sys_msg:
                f.write(f"**System**: {sys_msg}\n\n")
            f.write(f"**User**: {user_msg}\n\n")

            for run in data["runs"]:
                f.write(f"#### Run {run['run']} — {run['tokens']} tokens | {run['elapsed']}s\n\n")
                f.write(f"{run['output']}\n\n")
            f.write("---\n\n")

    # Print summary
    print(f"{'Category':<25} {'Mean words':>12}")
    print(f"{'-'*25} {'-'*12}")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        print(f"{cat:<25} {s['mean_words']:>12}")

    print(f"\nPer-prompt:")
    print(f"{'Prompt':<30} {'Cat':<20} {'Words':>6} {'Tokens':>7}")
    print(f"{'-'*30} {'-'*20} {'-'*6} {'-'*7}")
    for p in all_prompts:
        pid = p["id"]
        if pid in prompt_stats:
            ps = prompt_stats[pid]
            print(f"{pid:<30} {ps['category']:<20} {ps['mean_words']:>6} {ps['mean_tokens']:>7}")

    print(f"\nResults saved to:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
