
import json
from typing import List, Dict, Any
from torch.utils.data import Dataset

from labels import LABEL_TO_ID

class STTPIIDataset(Dataset):
    """
    Loads JSONL of the form:
    {"id": "utt_0001", "text": "my email is ...", "entities": [...]}
    Produces tokenized examples with BIO label ids aligned to tokenizer offsets.
    """
    def __init__(self, file_path: str, tokenizer, max_len: int = 256, training: bool = True):
        self._items: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.training = training
        self._load(file_path)

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                text = obj.get("text", "")
                entities = obj.get("entities", [])

                # Build character-level BIO tags
                char_tags = ["O"] * len(text)
                for ent in entities:
                    s, e, lab = ent["start"], ent["end"], ent["label"]
                    if not (0 <= s < e <= len(text)):
                        continue
                    char_tags[s] = f"B-{lab}"
                    for i in range(s + 1, e):
                        char_tags[i] = f"I-{lab}"

                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_len,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                offsets = enc["offset_mapping"]
                input_ids = enc["input_ids"]
                attn_mask = enc["attention_mask"]

                # Map offsets to BIO tags (use start index)
                token_bio = []
                for (st, ed) in offsets:
                    if st == ed:
                        # special token like [CLS], [SEP], or padding
                        token_bio.append("O")
                    else:
                        # pick char tag at start position
                        if st < len(char_tags):
                            token_bio.append(char_tags[st])
                        else:
                            token_bio.append("O")

                # Convert to ids
                label_ids = [LABEL_TO_ID.get(lb, LABEL_TO_ID["O"]) for lb in token_bio]
                # safety: align lengths
                if len(label_ids) != len(input_ids):
                    # fallback all O
                    label_ids = [LABEL_TO_ID["O"]] * len(input_ids)

                self._items.append({
                    "id": obj.get("id", ""),
                    "text": text,
                    "input_ids": input_ids,
                    "attention_mask": attn_mask,
                    "labels": label_ids,
                    "offsets": offsets,
                })

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def make_collate(pad_token_id: int, label_pad: int = -100):
    """
    Returns a collate_fn suitable for DataLoader.
    Pads input_ids, attention_mask, and label sequences to the max length of the batch.
    """
    def collate(batch):
        maxlen = max(len(ex["input_ids"]) for ex in batch)
        def pad_seq(seq, padv):
            return seq + [padv] * (maxlen - len(seq))

        input_ids = [pad_seq(item["input_ids"], pad_token_id) for item in batch]
        attention_mask = [pad_seq(item["attention_mask"], 0) for item in batch]
        labels = [pad_seq(item["labels"], label_pad) for item in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ids": [item["id"] for item in batch],
            "texts": [item["text"] for item in batch],
            "offsets": [item["offsets"] for item in batch],
        }
    return collate
