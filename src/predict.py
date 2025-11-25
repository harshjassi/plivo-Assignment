import argparse
import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import LABEL_TO_ID
from labels import LABELS, is_pii_label

# Rebuild id->label mapping locally to alter style
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}

# ---------- Validators for higher precision ----------
def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)

def looks_like_card(s: str) -> bool:
    d = _digits_only(s)
    return 13 <= len(d) <= 19

def looks_like_phone(s: str) -> bool:
    d = _digits_only(s)
    return 7 <= len(d) <= 15

def looks_like_email(s: str) -> bool:
    low = s.lower()
    # Accept both @/dot and ' at ' / ' dot ' spoken forms
    return ("@" in low and "." in low) or (" at " in low and " dot " in low)

def normalize_spoken_email(s: str) -> str:
    s = s.replace(" at ", "@").replace(" dot ", ".")
    return s

# ---------- BIO -> spans ----------
def decode_bio(offsets, label_ids):
    spans = []
    cur_label = None
    cur_start = None
    cur_end = None

    for (st, ed), lid in zip(offsets, label_ids):
        if st == 0 and ed == 0:
            # [CLS]/[SEP] or special token
            continue
        lab = ID2LABEL.get(int(lid), "O")
        if lab == "O":
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_label = None
            continue

        # label looks like B-FOO or I-FOO
        parts = lab.split("-", 1)
        if len(parts) != 2:
            # malformed
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_label = None
            continue
        tag, etype = parts
        if tag == "B":
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
            cur_label = etype
            cur_start = st
            cur_end = ed
        elif tag == "I":
            if cur_label == etype:
                cur_end = ed
            else:
                # I without B: start a new span conservatively
                if cur_label is not None:
                    spans.append((cur_start, cur_end, cur_label))
                cur_label = etype
                cur_start = st
                cur_end = ed

    if cur_label is not None:
        spans.append((cur_start, cur_end, cur_label))
    return spans

def filter_and_format_spans(text, spans):
    out = []
    for s, e, lab in spans:
        snippet = text[s:e]
        # conservative filters for high precision on PII
        if lab == "CREDIT_CARD":
            if not looks_like_card(snippet):
                continue
        if lab == "PHONE":
            if not looks_like_phone(snippet):
                continue
        if lab == "EMAIL":
            if not looks_like_email(snippet):
                continue
            snippet = normalize_spoken_email(snippet)

        out.append({
            "start": int(s),
            "end": int(e),
            "label": lab,
            "pii": bool(is_pii_label(lab))
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    predictions = {}

    with open(args.input, "r", encoding="utf-8") as fh:
        for raw in fh:
            obj = json.loads(raw)
            uid = obj.get("id", "")
            text = obj.get("text", "")

            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            offsets = enc["offset_mapping"].tolist()[0]
            input_ids = enc["input_ids"].to(args.device)
            attn = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[0]  # seq_len x num_labels
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = decode_bio(offsets, pred_ids)
            final_ents = filter_and_format_spans(text, spans)
            predictions[uid] = final_ents

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as of:
        json.dump(predictions, of, ensure_ascii=False, indent=2)

    print(f"Wrote {len(predictions)} predictions to {args.output}")

if __name__ == "__main__":
    main()
