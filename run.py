"""KLA submission entry point for restoring every NumPy image in a directory.

Example:
    python run.py /path/to/input_dir /path/to/output_dir
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "models" / "best_model.pt"


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    encoder_blocks: int = 2
    bottleneck_blocks: int = 4
    residual_scale: float = 0.1
    upscale: int = 2


class ResidualBlock(nn.Module):
    def __init__(self, width: int, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features + self.residual_scale * self.body(features)


def residual_stage(width: int, blocks: int, scale: float) -> nn.Sequential:
    return nn.Sequential(*(ResidualBlock(width, scale) for _ in range(blocks)))


class RestorationUNet(nn.Module):
    """Two-level residual U-Net with bicubic residual and PixelShuffle output."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.upscale != 2:
            raise ValueError("The submitted model supports exactly 2x restoration")

        width = config.width
        blocks = config.encoder_blocks
        scale = config.residual_scale
        self.upscale = config.upscale
        self.head = nn.Conv2d(1, width, 3, padding=1)
        self.encoder1 = residual_stage(width, blocks, scale)
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.encoder2 = residual_stage(width * 2, blocks, scale)
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.bottleneck = residual_stage(
            width * 4,
            config.bottleneck_blocks,
            scale,
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(width * 4, width * 8, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.fuse2 = nn.Conv2d(width * 4, width * 2, 1)
        self.decoder2 = residual_stage(width * 2, blocks, scale)
        self.up1 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.fuse1 = nn.Conv2d(width * 2, width, 1)
        self.decoder1 = residual_stage(width, blocks, scale)
        self.output = nn.Sequential(
            nn.Conv2d(width, config.upscale**2, 3, padding=1),
            nn.PixelShuffle(config.upscale),
        )

    def forward(self, noisy: Tensor) -> Tensor:
        baseline = F.interpolate(
            noisy,
            scale_factor=self.upscale,
            mode="bicubic",
            align_corners=False,
        )
        encoder1 = self.encoder1(self.head(noisy))
        encoder2 = self.encoder2(self.down1(encoder1))
        features = self.bottleneck(self.down2(encoder2))
        features = self.decoder2(
            self.fuse2(torch.cat((self.up2(features), encoder2), dim=1))
        )
        features = self.decoder1(
            self.fuse1(torch.cat((self.up1(features), encoder1), dim=1))
        )
        return baseline + self.output(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing degraded 128x128 .npy files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where restored 256x256 .npy files will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def load_input(path: Path) -> Tensor:
    array = np.load(path, allow_pickle=False)
    if array.shape != (128, 128):
        raise ValueError(f"Expected a 128x128 array at {path}, found {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Expected a numeric array at {path}, found {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"Input contains NaN or infinity: {path}")
    # Do not clip the input: speckle excursions outside [0, 1] contain signal.
    return torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise KeyError("Checkpoint must contain 'model' and 'model_config'")

    config = ModelConfig(**checkpoint["model_config"])
    model = RestorationUNet(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(memory_format=torch.channels_last)
    return model


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    input_paths = sorted(args.input_dir.glob("*.npy"))
    if not input_paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    written = 0
    with torch.inference_mode():
        for offset in range(0, len(input_paths), args.batch_size):
            batch_paths = input_paths[offset : offset + args.batch_size]
            batch = torch.stack([load_input(path) for path in batch_paths])
            batch = batch.to(device, non_blocking=True)
            if device.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)

            with autocast_context(device):
                restored = model(batch)
            restored = restored.float().clamp(0.0, 1.0).cpu().numpy()

            for input_path, image in zip(batch_paths, restored, strict=True):
                output = image[0].astype(np.float32, copy=False)
                if output.shape != (256, 256) or not np.isfinite(output).all():
                    raise RuntimeError(f"Invalid model output for {input_path.name}")
                np.save(args.output_dir / input_path.name, output)
                written += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Restored {written} images to {args.output_dir}")
    print(f"Total wall time: {elapsed:.3f} seconds")
    print(f"Average end-to-end time: {elapsed * 1000.0 / written:.3f} ms/image")


if __name__ == "__main__":
    main()
