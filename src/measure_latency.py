import argparse
import json
import statistics
import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def measure(model, tokenizer, texts, device, runs=50, maxlen=256):
    # Warmup passes
    for _ in range(5):
        t = texts[0]
        enc = tokenizer(t, truncation=True, max_length=maxlen, return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))

    times = []
    n = min(len(texts), runs)
    for i in range(runs):
        t = texts[i % len(texts)]
        enc = tokenizer(t, truncation=True, max_length=maxlen, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return times

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line.strip())
            texts.append(obj.get("text", ""))

    if not texts:
        print("No texts for latency measurement.")
        return

    times = measure(model, tokenizer, texts, args.device, runs=args.runs, maxlen=args.max_length)
    p50 = statistics.median(times)
    times_sorted = sorted(times)
    idx95 = max(0, min(len(times_sorted)-1, int(0.95 * len(times_sorted)) - 1))
    p95 = times_sorted[idx95]
    print(f"Latency over {len(times)} runs (batch_size=1): p50={p50:.2f} ms p95={p95:.2f} ms")

if __name__ == "__main__":
    main()
