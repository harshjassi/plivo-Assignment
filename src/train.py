import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import STTPIIDataset, make_collate
from model import build_token_classifier
from labels import LABELS

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--dev", default="data/dev.jsonl")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def run_training(args):
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.max_length)
    train_ds = STTPIIDataset(args.train, tokenizer, max_len=args.max_length, training=True)
    collate = make_collate(tokenizer.pad_token_id, label_pad=-100)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = build_token_classifier(args.model_name)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{args.epochs}")
        for batch in loop:
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=args.device)
            attn = torch.tensor(batch["attention_mask"], dtype=torch.long, device=args.device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # save artifacts
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model and tokenizer to {args.out_dir}")

if __name__ == "__main__":
    args = parse_args()
    run_training(args)
