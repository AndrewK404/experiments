# Running on RunPod

Use a **PyTorch 2.x** pod template: CUDA and torch are already installed. Work inside
`/workspace` -- it is the only directory that survives a pod restart.

## Install

```bash
cd /workspace
git clone https://github.com/<you>/simple-llm-train.git && cd simple-llm-train

# reuse the preinstalled torch (skips a ~2.5 GB CUDA wheel download)
pip install "einops>=0.8" "einx>=0.4" "jaxtyping>=0.3" python-dotenv "wandb>=0.29" pyarrow huggingface-hub
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Isolated alternative -- cleaner, but downloads torch again:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
export UV_CACHE_DIR=/workspace/.uv-cache      # keep the cache on the persistent volume
uv sync
```

## Keys and caches

```bash
printf 'WANDB_API_KEY=<your key>\nWANDB_MODE=online\n' > .env   # load_dotenv() picks this up
export HF_HOME=/workspace/.hf                                   # only needed if you log in to HF
                                                                # or later pull whole repos/models
```

## Data

`data/` is git-ignored, so prepare the corpus on the pod. `hf_data.py` streams parquet over range
requests and keeps no on-disk cache, so it re-downloads its slice on every run:

```bash
python data.py                                   # TinyShakespeare
python hf_data.py --limit 50000                  # or 50k TinyStories
```

## Train

`pick_device` selects `cuda` automatically -- no flag needed.

```bash
python train.py --profile smoke                  # ~10 s, verifies the whole pipeline
python train.py --profile default                # baseline, sized for CPU/MPS
```

The default profile (3.5M params) underuses a datacenter GPU. Scale it up:

```bash
python train.py \
  --model.d_model 512 --model.num_layers 8 --model.num_heads 8 --model.context_length 512 \
  --train.batch_size 64 --train.max_steps 20000 --train.warmup_steps 500 \
  --out_dir runs/gpu --wandb.name gpu
```

Once a configuration repeats, add it as a profile in `PROFILES` instead of keeping a wall of flags
in your shell history.

## Long runs

An SSH drop kills the process, so detach it:

```bash
tmux new -s train                                # detach with Ctrl+B, then D
python train.py --profile default 2>&1 | tee runs/train.log
```

Reattach with `tmux attach -t train`.

Checkpoints land in `runs/<name>/ckpt.pt`. On a pod with ephemeral storage everything outside
`/workspace` is lost on restart, so clone the repo there and resume with
`--train.resume runs/<name>/ckpt.pt --train.max_steps <higher>`.
