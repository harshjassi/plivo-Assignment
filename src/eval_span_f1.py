import argparse
import json
from collections import defaultdict
from labels import is_pii_label

def load_gold(path):
    d = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            uid = obj.get("id", "")
            spans = []
            for e in obj.get("entities", []):
                spans.append((e["start"], e["end"], e["label"]))
            d[uid] = spans
    return d

def load_pred(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    d = {}
    for uid, ents in data.items():
        spans = []
        for e in ents:
            spans.append((e["start"], e["end"], e["label"]))
        d[uid] = spans
    return d

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)

    labelset = set()
    for spans in gold.values():
        for _, _, lab in spans:
            labelset.add(lab)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # accumulate exact-match spans
    all_uids = set(list(gold.keys()) + list(pred.keys()))
    for uid in all_uids:
        g = set(gold.get(uid, []))
        p = set(pred.get(uid, []))

        for item in p:
            if item in g:
                tp[item[2]] += 1
            else:
                fp[item[2]] += 1
        for item in g:
            if item not in p:
                fn[item[2]] += 1

    # per-entity metrics
    print("Per-entity metrics:")
    macro_sum = 0.0
    count = 0
    for lab in sorted(labelset):
        p, r, f = prf(tp[lab], fp[lab], fn[lab])
        print(f"{lab:15s} P={p:.3f} R={r:.3f} F1={f:.3f}")
        macro_sum += f
        count += 1
    macro_f1 = macro_sum / max(1, count)
    print(f"\nMacro-F1: {macro_f1:.3f}")

    # PII / non-PII aggregated metrics
    pii_tp = pii_fp = pii_fn = 0
    non_tp = non_fp = non_fn = 0
    for uid in all_uids:
        g = gold.get(uid, [])
        p = pred.get(uid, [])

        g_pii = set((s, e) for s, e, l in g if is_pii_label(l))
        p_pii = set((s, e) for s, e, l in p if is_pii_label(l))
        g_non = set((s, e) for s, e, l in g if not is_pii_label(l))
        p_non = set((s, e) for s, e, l in p if not is_pii_label(l))

        for item in p_pii:
            if item in g_pii:
                pii_tp += 1
            else:
                pii_fp += 1
        for item in g_pii:
            if item not in p_pii:
                pii_fn += 1

        for item in p_non:
            if item in g_non:
                non_tp += 1
            else:
                non_fp += 1
        for item in g_non:
            if item not in p_non:
                non_fn += 1

    p, r, f = prf(pii_tp, pii_fp, pii_fn)
    print(f"\nPII metrics: P={p:.3f} R={r:.3f} F1={f:.3f}")
    p2, r2, f2 = prf(non_tp, non_fp, non_fn)
    print(f"Non-PII metrics: P={p2:.3f} R={r2:.3f} F1={f2:.3f}")

if __name__ == "__main__":
    main()
