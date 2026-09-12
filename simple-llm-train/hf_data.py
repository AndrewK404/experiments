"""Export a Hugging Face dataset to a flat .txt in data/raw/, ready for data.py.

    python hf_data.py                                                  # 50k TinyStories
    python hf_data.py --limit 5000
    python hf_data.py --dataset cardiffnlp/tweet_eval --config offensive --split train \
        --label 1 --separator "\n" --out data/raw/tweets.txt
    python hf_data.py --dataset Ananda100/python-clean-codeparrot --text-col content \
        --limit 100000 --ascii-only --separator "\n\n\n# ---\n" --out data/raw/python-code.txt

Then train on it:
    python train.py --data.raw_path data/raw/tinystories.txt \
        --data.out_dir data/processed/tinystories --train.prompt "STORY:"

Reads the Hub's parquet export over HTTP range requests, so only the rows you ask for are
downloaded -- TinyStories ships as 1.9 GB of text, and --limit 50000 pulls a few dozen MB.
"""
import argparse
import os

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

BATCH = 10_000


def fetch(args) -> list[str]:
    """First `limit` rows of the text column, optionally filtered by an integer label."""
    fs = HfFileSystem()
    pattern = f"datasets/{args.dataset}@refs/convert/parquet/{args.config}/{args.split}/*.parquet"
    files = sorted(fs.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet export found for {args.dataset} ({args.config}/{args.split})")

    columns = [args.text_col] + ([args.label_col] if args.label is not None else [])
    items: list[str] = []
    for path in files:
        with fs.open(path, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=BATCH, columns=columns):
                texts = batch.column(args.text_col).to_pylist()
                if args.label is not None:
                    labels = batch.column(args.label_col).to_pylist()
                    texts = [t for t, y in zip(texts, labels) if y == args.label]
                items += [t.strip() for t in texts if len(t.strip()) >= args.min_chars]
                if len(items) >= args.limit:
                    return items[: args.limit]
    return items


def main(args) -> None:
    items = fetch(args)
    # The separator prefixes every item, so a fresh document always starts the same way and
    # --train.prompt can reuse it as a sampling prompt.
    text = "".join(f"{args.separator}{t}" for t in items).lstrip("\n")
    if args.ascii_only:  # char-level: emoji and stray unicode inflate the vocab
        text = text.encode("ascii", "ignore").decode()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"{args.out}: {len(items):,} items, {len(text):,} chars, vocab {len(set(text))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--config", default="default", help="dataset config name, e.g. offensive for tweet_eval")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--label", type=int, default=None, help="keep only rows with this integer label")
    p.add_argument("--min-chars", type=int, default=1)
    # argparse does no escape processing, and "\n" in a shell argument is a literal backslash-n --
    # so decode escapes here instead of making every caller reach for $'...' quoting.
    p.add_argument("--separator", default="\n\n\nSTORY: ", type=lambda s: s.encode().decode("unicode_escape"),
                   help=r"prepended to every item; \n and \t are interpreted")
    p.add_argument("--ascii-only", action="store_true")
    p.add_argument("--out", default="data/raw/tinystories.txt")
    main(p.parse_args())
