# simple-llm-train

A char-level transformer language model with a small but honest training pipeline:
dataclass-based config with named profiles, W&B logging, checkpoints with resume, periodic
eval and text samples. No yaml, no orchestration layer.

```
config.py         dataclasses + profiles (default, smoke, gpu) + CLI; the single source of truth
hf_data.py        export a Hugging Face dataset to a flat txt in data/raw/ (TinyStories by default)
data.py           txt -> char tokens -> data/processed/{train,val}.npy + vocab.json
train.py          training loop, W&B, ckpt/resume, eval, samples, metrics.json
transformer/      model.py, optimizer.py, nn_utils.py, data.py
runs/<name>/      ckpt.pt, config.json, metrics.json   (git-ignored)
data/             raw txt and .npy token arrays        (git-ignored)
```

## Setup

```bash
uv sync
cp .env.example .env             # then put your WANDB_API_KEY in it

uv run python hf_data.py --limit 20000        # the default corpus
uv run python train.py                        # train on it
```

Without a key, run with `--wandb.mode disabled` (or `offline` to keep local logs only).

## Data

The default corpus is TinyStories, written by `hf_data.py`. It reads the Hub's parquet export over
HTTP range requests, so only the rows you ask for are downloaded (TinyStories ships as 1.9 GB of
text; `--limit 50000` pulls a few dozen MB):

```bash
uv run python hf_data.py --limit 20000                   # -> data/raw/tinystories.txt, ~18M chars
uv run python hf_data.py --limit 550000                  # ~500M chars, what the gpu profile wants

uv run python hf_data.py --dataset cardiffnlp/tweet_eval --config offensive \
    --label 1 --separator $'\n' --out data/raw/tweets.txt
```

`--separator` (default `\n\n\nSTORY: `) is prepended to every item, so each document starts the
same way and the marker doubles as a sampling prompt -- hence `train.prompt = "STORY:"`.

Tokenization into `data/processed/` happens automatically on the first training run. Do it
explicitly to inspect the corpus first, or to use a different one:

```bash
uv run python data.py                                          # tokenize data/raw/tinystories.txt
uv run python data.py --profile smoke                          # downloads TinyShakespeare instead
uv run python data.py --data.url <raw url> --data.raw_path data/raw/mine.txt
```

The tokenizer is built from the text itself, so the vocabulary is whatever characters the corpus
contains -- `hf_data.py --ascii-only` keeps it small. `--data.val_fraction 0.1` controls the tail
held out for validation.

Note that windows are sampled from one flat token stream, so a batch can straddle two documents --
there is no document-level attention masking.

GPU pods: see [runpod_setup.md](runpod_setup.md).

## Training

```bash
uv run python train.py --profile smoke        # 30 steps on TinyShakespeare, ~10 s, W&B disabled
uv run python train.py                        # default profile: ~3.5M params, 1000 steps
uv run python train.py --profile gpu          # ~25M params on TinyStories, sized for a 4090
uv run python train.py --help                 # every field, showing the current profile's values
```

`train.amp` selects mixed precision on CUDA: `bf16` (default) or `off`. bf16 keeps fp32's exponent
range, so no loss scaling is involved. CPU and MPS always run fp32. The loss stays in fp32 either way,
since softmax and cross-entropy here are hand-written rather than autocast-aware ATen ops.

Each eval also generates a sample, one token at a time -- `train.sample_tokens` (default 300)
bounds that cost, and `train.eval_every` controls how often you pay it. Sampling past
`context_length` works: `generate()` slides its window, so the model keeps writing and simply
forgets what fell out of that window.

Any config field is a flag, so an experiment does not need a code change:

```bash
uv run python train.py --train.lr 3e-4 --model.d_model 512
uv run python train.py --profile smoke --wandb.mode online --train.max_steps 100
uv run python train.py --model.rope_theta null --train.betas "[0.9, 0.99]"
```

Values are parsed with `ast.literal_eval`, so `3e-4`, `null` and `[0.9, 0.99]` arrive as
`float`, `None` and `list`. Overrides are applied on top of the profile, then the config is
rebuilt -- so validation and derived fields also run on CLI values.

Resume from a checkpoint, raising the step budget (the step counter comes from the checkpoint, so
`max_steps` must exceed it or the loop exits immediately):

```bash
uv run python train.py --train.resume runs/default/ckpt.pt --train.max_steps 2000
```

Each run writes `runs/<name>/{ckpt.pt,config.json,metrics.json}`. `config.json` carries the git
sha, and `ckpt.pt` embeds the full config, so a run is reproducible from its artifacts rather
than from shell history.

## Profiles

A profile is a function returning a `Config`:

```python
def smoke() -> Config:
    return Config(
        out_dir="runs/smoke",
        model=ModelCfg(context_length=64, d_model=64, num_layers=2, num_heads=2),
        train=TrainCfg(max_steps=30, warmup_steps=5, ...),
        wandb=WandbCfg(name="smoke", mode="disabled"),
    )

PROFILES = {"default": default, "smoke": smoke}
```

A new experiment means a new function plus one line in `PROFILES`, never an edit to the defaults --
editing defaults silently makes past runs irreproducible.

Two things the config does beyond holding values:

- **Derived fields.** `d_ff = 8/3 * d_model`, rounded to a multiple of 64, is computed in
  `__post_init__`. Such fields are listed in `DERIVED` so `--model.d_model 512` recomputes
  `d_ff` (1344) instead of keeping the profile's value, while an explicit `--model.d_ff 1024` wins.
- **Validation up front.** `d_model % num_heads` and `warmup_steps < max_steps` fail in the first
  second, not on step 900.

## W&B

Provide the key via `.env` (`WANDB_API_KEY`), `wandb login`, or the environment. Logged per run:
`train/loss`, `val/loss`, `lr`, `step_time`, a text sample at every eval (one incremental table),
and the checkpoint as an artifact.

Run names carry a to-the-second timestamp: `wandb.name = "gpu"` shows up as
`gpu 2026-09-10 19:52:01`, and leaving it `null` names the run by timestamp alone -- reruns of the
same profile stay distinguishable instead of collapsing into one name.

`wandb.mode` accepts `online | offline | disabled`. Leaving it `null` defers to `WANDB_MODE` from
the environment, which is why the config default is `None` -- an explicit value in code would
override the environment. The `smoke` profile pins `disabled` so pipeline checks do not clutter
the project. An `offline` run can be uploaded later with `wandb sync wandb/offline-run-*`.

## Notes

Verified end-to-end: `--profile smoke` on Apple MPS (loss 4.31 -> 3.15 over 30 steps) and resume
from a checkpoint.

When the configuration space grows past two or three axes (model x data x hardware), this is the
point where yaml profiles or Hydra composition start to pay off; until then a typed python config
is shorter and gives autocompletion for free.
