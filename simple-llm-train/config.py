"""Config: dataclasses + named profiles. Single source of truth, no yaml.

    python train.py                                  # default profile
    python train.py --profile smoke                  # quick run, ~10 s on CPU
    python train.py --train.lr 3e-4 --model.d_model 512
    python train.py --profile smoke --wandb.mode online
    python train.py --help                           # every field with its profile value

A new experiment means a new function in PROFILES (see smoke), not an edit to the defaults.
"""
import argparse
import ast
import json
import subprocess
from dataclasses import asdict, dataclass, field

# Fields computed in __post_init__: a CLI override of the parent must recompute them.
DERIVED = ("model.d_ff",)


# A small plain-text corpus to point --data.url at when you want a one-file download instead of hf_data.py.
TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class DataCfg:
    raw_path: str = "data/raw/tinystories.txt"  # produced by hf_data.py
    out_dir: str = "data/processed/tinystories"
    url: str | None = None  # if set, raw_path is downloaded from here when missing
    val_fraction: float = 0.1


@dataclass
class ModelCfg:
    context_length: int = 256
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int | None = None  # None -> 8/3 * d_model, rounded to a multiple of 64 (SwiGLU)
    rope_theta: float | None = 10000.0

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = round(8 * self.d_model / 3 / 64) * 64
        if self.d_model % self.num_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by num_heads={self.num_heads}")


@dataclass
class TrainCfg:
    batch_size: int = 32
    max_steps: int = 1000
    lr: float = 1.0e-3
    min_lr: float = 1.0e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 100
    eval_dense_until: int = 0  # below this step eval runs every eval_dense_every instead; 0 disables
    eval_dense_every: int = 200
    eval_steps: int = 20
    ckpt_every: int = 500
    resume: str | None = None  # path to ckpt.pt
    prompt: str = "ROMEO:"  # sampling prompt; "STORY:" for the TinyStories corpus from hf_data.py
    sample_tokens: int = 300  # tokens generated at each eval; they are produced one by one, so this costs time
    amp: str = "bf16"  # bf16 | off -- mixed precision, CUDA only (CPU/MPS always run fp32)

    def __post_init__(self):
        if self.warmup_steps >= self.max_steps:
            raise ValueError(f"warmup_steps={self.warmup_steps} >= max_steps={self.max_steps}")
        if self.amp not in ("bf16", "off"):
            raise ValueError(f"amp={self.amp!r}, expected bf16 | off")
        if self.eval_dense_until and self.eval_dense_every <= 0:
            raise ValueError(f"eval_dense_every={self.eval_dense_every}, expected > 0 when eval_dense_until is set")


@dataclass
class WandbCfg:
    project: str = "simple-llm-train"
    name: str | None = None
    mode: str | None = None  # online | offline | disabled; None -> env WANDB_MODE, else online


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | mps
    out_dir: str = "runs/default"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Inverse of to_dict() -- e.g. from ckpt["config"]."""
        return cls(
            **{
                **d,
                "data": DataCfg(**d["data"]),
                "model": ModelCfg(**d["model"]),
                "train": TrainCfg(**d["train"]),
                "wandb": WandbCfg(**d["wandb"]),
            }
        )

    def save(self, path: str) -> None:
        """Config next to the checkpoint: a run is repeatable from the artifact, not from memory."""
        with open(path, "w") as f:
            json.dump({**self.to_dict(), "git_sha": git_sha()}, f, indent=2)


# ---------------------------------------------------------------- profiles


def default() -> Config:
    """Baseline run: ~3.5M parameters, 1000 steps. Needs `python hf_data.py --limit 20000` first."""
    return Config(train=TrainCfg(prompt="STORY:"))


def smoke() -> Config:
    """Tiny run to exercise the whole pipeline (~10 s on CPU, W&B off)."""
    return Config(
        seed=0,
        out_dir="runs/smoke",
        data=DataCfg(raw_path="data/raw/tinyshakespeare.txt", out_dir="data/processed/smoke", url=TINY_SHAKESPEARE),
        model=ModelCfg(context_length=64, d_model=64, num_layers=2, num_heads=2),
        train=TrainCfg(
            batch_size=8,
            max_steps=30,
            warmup_steps=5,
            log_every=5,
            eval_every=15,
            eval_steps=2,
            ckpt_every=30,
            sample_tokens=100,
        ),
        wandb=WandbCfg(name="smoke", mode="disabled"),
    )


def gpu() -> Config:
    """~25M params on TinyStories, sized for about an hour on an RTX 4090.

    Run `python hf_data.py --limit 550000` first -- this profile expects that corpus.
    Calibrate max_steps from the measured step_time: max_steps ~= 2700 / step_time.
    """
    return Config(
        out_dir="runs/gpu",
        model=ModelCfg(context_length=256, d_model=512, num_layers=8, num_heads=8),
        train=TrainCfg(
            batch_size=64,
            max_steps=20000,
            warmup_steps=400,
            weight_decay=0.1,
            log_every=50,
            eval_every=1000,
            eval_steps=20,
            ckpt_every=2000,
            prompt="STORY:",
            sample_tokens=300,
        ),
        wandb=WandbCfg(name="gpu"),
    )


def code() -> Config:
    r"""~25M params on Python source, sized for about 50 minutes on an RTX 4090.

    Run this first (--ascii-only matters: unicode in comments would bloat a char-level vocab):
        python hf_data.py --dataset Ananda100/python-clean-codeparrot --text-col content \
            --limit 100000 --ascii-only --separator "\n\n\n# ---\n" --out data/raw/python-code.txt

    100k files is ~520M characters: one 20k-step run reads 64*256*20000 = 328M, so that is a bit
    over one epoch, with room for a longer calibrated run. Going much higher mostly costs RAM --
    data.py holds the whole corpus as a Python list while tokenizing (~16 GB at 200k files).

    Calibrate max_steps from the measured step_time: max_steps ~= 3000 / step_time.

    The prompt opens both loops of a training loop, so the very next thing the sample owes us is the
    body at twelve spaces: zero_grad / forward / loss / backward / step. That is a checklist you can
    grade against, unlike "STORY:", which looked plausible almost immediately and then said nothing
    about progress.

    sample_tokens is 1000 against a context_length of 256 on purpose. The prompt scrolls out of the
    window after the first ~256 generated characters, so the tail measures something else entirely:
    whether the model keeps writing coherent Python with no prompt left to lean on.
    """
    return Config(
        out_dir="runs/code",
        data=DataCfg(raw_path="data/raw/python-code.txt", out_dir="data/processed/python-code"),
        model=ModelCfg(context_length=256, d_model=512, num_layers=8, num_heads=8),
        train=TrainCfg(
            batch_size=64,
            max_steps=20000,
            warmup_steps=400,
            weight_decay=0.1,
            log_every=50,
            eval_every=1000,
            eval_dense_until=1000,  # 0, 200, 400, 600, 800, then every 1000
            eval_dense_every=200,
            eval_steps=20,
            ckpt_every=2000,
            prompt=(
                "def train(model, loader, optimizer):\n"
                "    for epoch in range(10):\n"
                "        for x, y in loader:\n"
            ),
            sample_tokens=1000,  # 4x the context window: the tail is written with the prompt long gone
        ),
        wandb=WandbCfg(name="code"),
    )


PROFILES = {"default": default, "smoke": smoke, "gpu": gpu, "code": code}


# ---------------------------------------------------------------- CLI


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _flat(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def _set(d: dict, key: str, value) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        d = d[p]
    d[leaf] = value


def _parse(s: str):
    """'3e-4' -> float, 'null' -> None, '[0.9, 0.95]' -> list, anything else -> str."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {"true": True, "false": False, "null": None, "none": None}.get(s.lower(), s)


def parse_cli(argv=None) -> Config:
    """Profile + override of any field: --train.lr 3e-4 --model.d_model 512."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", default="default", choices=list(PROFILES), help="default: default")
    args, rest = pre.parse_known_args(argv)

    d = PROFILES[args.profile]().to_dict()
    p = argparse.ArgumentParser(parents=[pre], description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in _flat(d):
        p.add_argument(f"--{k}", type=_parse, default=argparse.SUPPRESS, help=f"{args.profile}: {v}")
    overrides = {k: v for k, v in vars(p.parse_args(rest)).items() if k != "profile"}

    for k in DERIVED:  # with --model.d_model 512, d_ff must be recomputed instead of kept from the profile
        if k not in overrides:
            _set(d, k, None)
    for k, v in overrides.items():
        _set(d, k, v)
    return Config.from_dict(d)  # __post_init__ = validation and derived fields
