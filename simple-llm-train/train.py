"""Training: loop + W&B + checkpoint/resume + eval + sample.

    python train.py --profile smoke                  # exercise the pipeline
    python train.py                                  # real run
    python train.py --train.lr 3e-4 --model.d_model 512
    python train.py --train.resume runs/default/ckpt.pt --train.max_steps 2000
"""
import json
import os
import time

import torch
import wandb
from dotenv import load_dotenv

import data
from config import Config, git_sha, parse_cli
from transformer.data import get_batch
from transformer.model import BasicsTransformerLM
from transformer.nn_utils import clip_gradient, cross_entropy
from transformer.optimizer import AdamW, get_cosine_lr


def amp_ctx(cfg: Config, device: str):
    """bf16 autocast on CUDA; a no-op elsewhere -- CPU and MPS stay in fp32."""
    on = cfg.train.amp == "bf16" and device.startswith("cuda")
    return torch.autocast(device_type="cuda" if on else "cpu", dtype=torch.bfloat16, enabled=on)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def build(cfg: Config, vocab_size: int, device: str):
    torch.manual_seed(cfg.seed)
    model = BasicsTransformerLM(vocab_size=vocab_size, **vars(cfg.model)).to(device)
    opt = AdamW(model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.betas), weight_decay=cfg.train.weight_decay)
    step, run_id, best_val = 0, None, float("inf")
    if cfg.train.resume:
        ckpt = torch.load(cfg.train.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        run_id = ckpt.get("run_id")  # .get: checkpoints written before these fields existed still load
        best_val = ckpt.get("best_val", float("inf"))
    return model, opt, step, run_id, best_val


@torch.no_grad()
def evaluate(model, ds, cfg: Config, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.train.eval_steps):
        x, y = get_batch(ds, cfg.train.batch_size, cfg.model.context_length, device)
        with amp_ctx(cfg, device):
            logits = model(x)
        losses.append(cross_entropy(logits.float(), y).item())
    model.train()
    return sum(losses) / len(losses)


def train_step(model, opt, ds, cfg: Config, device: str, step: int) -> tuple[float, float]:
    lr = get_cosine_lr(step, cfg.train.lr, cfg.train.min_lr, cfg.train.warmup_steps, cfg.train.max_steps)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch(ds, cfg.train.batch_size, cfg.model.context_length, device)
    with amp_ctx(cfg, device):
        logits = model(x)
    loss = cross_entropy(logits.float(), y)  # softmax/CE are hand-written, so keep them in fp32
    opt.zero_grad(set_to_none=True)
    loss.backward()
    clip_gradient(model.parameters(), cfg.train.grad_clip)
    opt.step()
    return loss.item(), lr


def should_eval(step: int, cfg: Config) -> bool:
    """Dense evals over the first eval_dense_until steps -- that is where the samples change from
    noise to syntax, and a coarse eval_every would step straight over it."""
    if step == cfg.train.max_steps:
        return True
    every = cfg.train.eval_dense_every if step < cfg.train.eval_dense_until else cfg.train.eval_every
    return step % every == 0


def save_ckpt(model, opt, step: int, cfg: Config, path: str, run_id: str | None, best_val: float) -> None:
    """Weights + optimizer + step + config + run identity: the whole run resumes from this one file."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "config": cfg.to_dict(),
            "run_id": run_id,  # so a resumed run continues the same W&B run instead of starting a new one
            "best_val": best_val,
        },
        path,
    )


@torch.no_grad()
def sample(model, tok, device: str, prompt: str = "ROMEO:", max_new_tokens: int = 300, cfg: Config | None = None) -> str:
    """Tokens are generated one at a time, so this is the expensive half of an eval -- keep it short."""
    x = torch.tensor(tok.encode(prompt), device=device)
    model.eval()
    with amp_ctx(cfg, device) if cfg else torch.autocast("cpu", enabled=False):
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8)
    model.train()
    return prompt + tok.decode(out[0].tolist())


def run_name(cfg: Config) -> str:
    """Run name with a to-the-second timestamp, so repeated runs stay distinguishable in W&B."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{cfg.wandb.name} {ts}" if cfg.wandb.name else ts


def train(cfg: Config) -> dict:
    device = pick_device(cfg.device)
    torch.backends.cuda.matmul.allow_tf32 = True  # matters when amp is off; free on Ampere and newer
    train_ds, val_ds, tok = data.load(cfg.data)
    model, opt, step, run_id, best_val = build(cfg, tok.vocab_size, device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(f"{cfg.out_dir}/config.json")
    ckpt_path, best_path = f"{cfg.out_dir}/ckpt.pt", f"{cfg.out_dir}/best.pt"
    run = wandb.init(
        project=cfg.wandb.project,
        name=None if run_id else run_name(cfg),  # a resumed run keeps the name it already has
        id=run_id,
        resume="allow" if run_id else None,
        mode=cfg.wandb.mode,
        config={**cfg.to_dict(), "runtime": {"device": device, "git_sha": git_sha(), "vocab_size": tok.vocab_size}},
    )
    amp = cfg.train.amp if device.startswith("cuda") else "off"
    print(f"device={device} amp={amp} params={model.get_num_params() / 1e6:.1f}M steps={step}->{cfg.train.max_steps}")

    # MUTABLE: every log re-sends the whole table, so run.summary["sample"] holds all the rows.
    # INCREMENTAL would upload only new rows, but then the summary points at a single increment.
    samples = wandb.Table(columns=["step", "val_loss", "text"], log_mode="MUTABLE")

    def log_eval(step: int) -> float:
        """Val loss + a sample at this step, to W&B and to stdout."""
        val_loss = evaluate(model, val_ds, cfg, device)
        text = sample(model, tok, device, cfg.train.prompt, cfg.train.sample_tokens, cfg)
        samples.add_data(step, val_loss, text)
        run.log({"val/loss": val_loss, "sample": samples}, step=step)
        print(f"step {step} val_loss {val_loss:.4f}\n{text}\n")
        return val_loss

    # Step 0 is the untrained baseline: val_loss should land near ln(vocab_size), and the sample
    # shows what "no training at all" looks like for this prompt.
    val_loss, loss = (log_eval(step) if step == 0 else float("nan")), float("nan")
    t0 = time.time()
    while step < cfg.train.max_steps:
        loss, lr = train_step(model, opt, train_ds, cfg, device, step)
        step += 1
        if step % cfg.train.log_every == 0:
            run.log({"train/loss": loss, "lr": lr, "step_time": (time.time() - t0) / cfg.train.log_every}, step=step)
            print(f"step {step} loss {loss:.4f} lr {lr:.2e} ({time.time() - t0:.1f}s)")
            t0 = time.time()
        if should_eval(step, cfg):
            val_loss = log_eval(step)
            if val_loss < best_val:
                # ckpt.pt is the latest state and a diverging run will overwrite it; best.pt is the
                # one that survives, and best_val rides along so a resume cannot clobber it either.
                best_val = val_loss
                save_ckpt(model, opt, step, cfg, best_path, run.id, best_val)
            t0 = time.time()  # eval and generation must not leak into the next step_time
        if step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps:
            save_ckpt(model, opt, step, cfg, ckpt_path, run.id, best_val)

    metrics = {"val_loss": val_loss, "train_loss": loss, "steps": step, "params_M": model.get_num_params() / 1e6}
    with open(f"{cfg.out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    art = wandb.Artifact("model", type="model", metadata=metrics)
    art.add_file(ckpt_path)
    if os.path.exists(best_path):
        art.add_file(best_path)
    run.log_artifact(art)
    run.finish()
    return metrics


if __name__ == "__main__":
    load_dotenv()  # WANDB_API_KEY from .env
    train(parse_cli())
