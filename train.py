"""Reproduce training of the submitted residual U-Net.

Expected dataset layout:
    TRAIN_DIR/
      NoisyLR/*.npy   # 128x128 degraded inputs
      GT/*.npy        # matching 256x256 clean targets

Example:
    python train.py /path/to/train --checkpoint models/best_model.pt
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import math
import random
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from evaluate import ModelConfig, RestorationUNet


SPLIT_SEED = 2026
VALIDATION_SIZE = 320


class PairedNpyDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        *,
        augment: bool,
    ) -> None:
        self.pairs = list(pairs)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        noisy_path, target_path = self.pairs[index]
        noisy = np.load(noisy_path, allow_pickle=False)
        target = np.load(target_path, allow_pickle=False)
        if noisy.shape != (128, 128) or target.shape != (256, 256):
            raise ValueError(
                f"Unexpected shapes for {noisy_path.name}: "
                f"input={noisy.shape}, target={target.shape}"
            )
        if not np.isfinite(noisy).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite values found in pair {noisy_path.name}")

        noisy_tensor = torch.from_numpy(
            noisy.astype(np.float32, copy=False)
        ).unsqueeze(0)
        target_tensor = torch.from_numpy(
            target.astype(np.float32, copy=False)
        ).unsqueeze(0)

        if self.augment:
            rotations = int(torch.randint(0, 4, ()).item())
            noisy_tensor = torch.rot90(noisy_tensor, rotations, dims=(-2, -1))
            target_tensor = torch.rot90(target_tensor, rotations, dims=(-2, -1))
            if bool(torch.randint(0, 2, ()).item()):
                noisy_tensor = torch.flip(noisy_tensor, dims=(-1,))
                target_tensor = torch.flip(target_tensor, dims=(-1,))

        return noisy_tensor.contiguous(), target_tensor.contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_dir",
        type=Path,
        help="Directory containing paired NoisyLR and GT subdirectories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/best_model.pt"),
    )
    parser.add_argument("--training-seconds", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--peak-lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def paired_paths(train_dir: Path) -> list[tuple[Path, Path]]:
    input_dir = train_dir / "NoisyLR"
    target_dir = train_dir / "GT"
    inputs = {path.name: path for path in input_dir.glob("*.npy")}
    targets = {path.name: path for path in target_dir.glob("*.npy")}
    if not inputs or not targets:
        raise FileNotFoundError(
            f"Expected .npy files in {input_dir} and {target_dir}"
        )
    missing_targets = sorted(inputs.keys() - targets.keys())
    missing_inputs = sorted(targets.keys() - inputs.keys())
    if missing_targets or missing_inputs:
        raise ValueError(
            "Input/target filenames do not match. "
            f"Missing targets: {missing_targets[:5]}; "
            f"missing inputs: {missing_inputs[:5]}"
        )
    return [(inputs[name], targets[name]) for name in sorted(inputs)]


def fixed_split(train_dir: Path) -> tuple[list, list]:
    pairs = paired_paths(train_dir)
    if len(pairs) <= VALIDATION_SIZE:
        raise ValueError(
            f"Need more than {VALIDATION_SIZE} pairs, found {len(pairs)}"
        )
    shuffled = pairs.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    validation_pairs = sorted(shuffled[:VALIDATION_SIZE])
    training_pairs = sorted(shuffled[VALIDATION_SIZE:])
    return training_pairs, validation_pairs


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    train_dir: Path,
    batch_size: int,
    workers: int,
    use_cuda: bool,
) -> tuple[DataLoader, DataLoader]:
    training_pairs, validation_pairs = fixed_split(train_dir)
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    common = {
        "num_workers": workers,
        "pin_memory": use_cuda,
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        PairedNpyDataset(training_pairs, augment=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        PairedNpyDataset(validation_pairs, augment=False),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")
    return train_loader, validation_loader


def structural_similarity(prediction: Tensor, target: Tensor) -> Tensor:
    window_size = 11
    sigma = 1.5
    coordinates = torch.arange(
        window_size,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    coordinates = coordinates - window_size // 2
    kernel_1d = torch.exp(-coordinates.square() / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).expand(
        prediction.shape[1],
        1,
        -1,
        -1,
    )
    padding = window_size // 2
    prediction = F.pad(prediction, (padding,) * 4, mode="reflect")
    target = F.pad(target, (padding,) * 4, mode="reflect")
    groups = prediction.shape[1]

    mean_prediction = F.conv2d(prediction, kernel, groups=groups)
    mean_target = F.conv2d(target, kernel, groups=groups)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_prediction * mean_target
    variance_prediction = (
        F.conv2d(prediction.square(), kernel, groups=groups) - mean_prediction_sq
    ).clamp_min(0.0)
    variance_target = (
        F.conv2d(target.square(), kernel, groups=groups) - mean_target_sq
    ).clamp_min(0.0)
    covariance = (
        F.conv2d(prediction * target, kernel, groups=groups) - mean_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mean_product + c1) * (2 * covariance + c2)
    denominator = (mean_prediction_sq + mean_target_sq + c1) * (
        variance_prediction + variance_target + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean(dim=(1, 2, 3))


def charbonnier(prediction: Tensor, target: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def restoration_loss(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    edge_loss = charbonnier(prediction_x, target_x) + charbonnier(
        prediction_y,
        target_y,
    )
    ssim = structural_similarity(prediction.float(), target.float()).mean()
    return charbonnier(prediction, target) + 0.10 * edge_loss + 0.10 * (1 - ssim)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available")
    return torch.device(name)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def learning_rate(progress: float, peak_lr: float) -> float:
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        return peak_lr * progress / warmup_fraction
    cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
    multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return peak_lr * multiplier


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    ssim_total = 0.0
    image_count = 0
    for noisy, target in loader:
        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if device.type == "cuda":
            noisy = noisy.contiguous(memory_format=torch.channels_last)
            target = target.contiguous(memory_format=torch.channels_last)
        with autocast_context(device):
            prediction = model(noisy)
        prediction = prediction.float().clamp(0.0, 1.0)
        target = target.float()
        squared_error += F.mse_loss(prediction, target, reduction="sum").item()
        pixel_count += target.numel()
        ssim_total += structural_similarity(prediction, target).sum().item()
        image_count += target.shape[0]
    mse = max(squared_error / pixel_count, 1e-12)
    return {
        "psnr": -10.0 * float(np.log10(mse)),
        "ssim": ssim_total / image_count,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.training_seconds <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Training time and batch size must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = ModelConfig()
    model = RestorationUNet(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    train_loader, validation_loader = make_loaders(
        args.train_dir,
        args.batch_size,
        args.workers,
        device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    steps = 0
    examples_seen = 0
    latest_loss = float("nan")
    train_iterator = iter(train_loader)
    started = time.perf_counter()
    model.train()
    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= args.training_seconds:
            break
        progress = min(elapsed / args.training_seconds, 1.0)
        try:
            noisy, target = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            noisy, target = next(train_iterator)

        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if device.type == "cuda":
            noisy = noisy.contiguous(memory_format=torch.channels_last)
            target = target.contiguous(memory_format=torch.channels_last)
        current_lr = learning_rate(progress, args.peak_lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            prediction = model(noisy)
            loss = restoration_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        steps += 1
        examples_seen += noisy.shape[0]
        latest_loss = loss.detach().item()
        if steps % 100 == 0:
            print(
                f"step={steps} loss={latest_loss:.6f} "
                f"lr={current_lr:.2e} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    metrics = evaluate(model, validation_loader, device)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(config),
            "training_config": {
                "batch_size": args.batch_size,
                "peak_lr": args.peak_lr,
                "weight_decay": args.weight_decay,
                "training_seconds": args.training_seconds,
                "seed": args.seed,
                "split_seed": SPLIT_SEED,
                "validation_size": VALIDATION_SIZE,
            },
            "metrics": metrics,
            "steps": steps,
        },
        args.checkpoint,
    )
    print("---")
    print(f"device:           {device}")
    print(f"val_psnr:         {metrics['psnr']:.6f}")
    print(f"val_ssim:         {metrics['ssim']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"num_steps:        {steps}")
    print(f"examples_seen:    {examples_seen}")
    print(f"final_loss:       {latest_loss:.6f}")
    print(f"checkpoint:       {args.checkpoint}")


if __name__ == "__main__":
    main()
