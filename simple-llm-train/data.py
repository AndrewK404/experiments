"""Data: txt -> char-level tokens -> data/processed/{train,val}.npy + vocab.json

    python hf_data.py                 # first: write data/raw/tinystories.txt
    python data.py                    # then tokenize it (train.py does this on its own too)
    python data.py --profile smoke    # downloads TinyShakespeare instead, no HF needed
    python data.py --data.url <raw url of any .txt> --data.raw_path data/raw/mine.txt
"""
import json
import os
import urllib.request

import numpy as np

from config import DataCfg


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            return cls(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.chars, f)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def prepare(cfg: DataCfg) -> None:
    if not os.path.exists(cfg.raw_path):
        if cfg.url is None:
            raise SystemExit(
                f"{cfg.raw_path} is missing and data.url is not set.\n"
                f"Write the corpus first (e.g. `python hf_data.py --limit 20000 --out {cfg.raw_path}`), "
                f"or point --data.url at a plain-text file to download."
            )
        os.makedirs(os.path.dirname(cfg.raw_path), exist_ok=True)
        urllib.request.urlretrieve(cfg.url, cfg.raw_path)
    with open(cfg.raw_path) as f:
        text = f.read()
    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.uint16)
    n_train = int(len(ids) * (1 - cfg.val_fraction))
    os.makedirs(cfg.out_dir, exist_ok=True)
    tok.save(f"{cfg.out_dir}/vocab.json")
    np.save(f"{cfg.out_dir}/train.npy", ids[:n_train])
    np.save(f"{cfg.out_dir}/val.npy", ids[n_train:])
    print(f"{len(text):,} chars, vocab {tok.vocab_size} -> train {n_train:,} / val {len(ids) - n_train:,} tokens")


def load(cfg: DataCfg) -> tuple[np.ndarray, np.ndarray, CharTokenizer]:
    """Prepares the data if it is missing, then returns train/val memmaps + the tokenizer."""
    if not os.path.exists(f"{cfg.out_dir}/train.npy"):
        prepare(cfg)
    train = np.load(f"{cfg.out_dir}/train.npy", mmap_mode="r")
    val = np.load(f"{cfg.out_dir}/val.npy", mmap_mode="r")
    return train, val, CharTokenizer.load(f"{cfg.out_dir}/vocab.json")


if __name__ == "__main__":
    from config import parse_cli

    prepare(parse_cli().data)
