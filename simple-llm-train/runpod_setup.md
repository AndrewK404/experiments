# Running on RunPod

Use a **PyTorch 2.x** pod template: CUDA and torch are already installed. Work inside
`/workspace` -- it is the only directory that survives a pod restart.

## Install

```bash
cd /workspace
git clone https://github.com/andrewk404/simple-llm-train.git && cd simple-llm-train

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
python hf_data.py --limit 550000                 # TinyStories, ~500M chars, what --profile gpu wants
python hf_data.py --limit 20000                  # a smaller slice to try things out
```

## Train

`pick_device` selects `cuda` automatically -- no flag needed.

```bash
python train.py --profile smoke                  # ~10 s, verifies the whole pipeline
python train.py --profile gpu                    # ~25M params, bf16, sized for an RTX 4090
```

Calibrate `max_steps` before the real run: train 100 steps, read `step_time` from the log, then
set `max_steps ~= (seconds you want to spend - 300) / step_time`.

```bash
python train.py --profile gpu --train.max_steps 100 --train.warmup_steps 10 \
  --wandb.mode disabled --out_dir runs/calib
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
