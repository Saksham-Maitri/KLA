# Efficient Residual U-Net for Semiconductor Image Restoration

This repository contains our submission for **AI-Based Restoration of Degraded
Images for Semiconductor Inspection**. The model jointly removes speckle and
Gaussian degradation while performing 2x grayscale super-resolution.

The submitted model is a compact residual U-Net with a bicubic skip path and
PixelShuffle reconstruction. It accepts degraded `128x128` NumPy arrays and
writes restored `256x256` NumPy arrays with matching filenames.

## Validation Results

Results use a deterministic 90/10 split of the 3,200 paired training images:

| Metric | Value |
|---|---:|
| PSNR | 26.952372 dB |
| SSIM | 0.782151 |
| Model-only batch-1 latency | 1.253974 ms/image |
| Parameters | 2.095 million |
| Peak training VRAM | 570.6 MB |
| Checkpoint size | 8.1 MiB |

The latency measurement was taken on the experiment GPU and excludes file I/O.

## Repository Layout

```text
.
├── README.md
├── evaluate.py
├── train.py
├── requirements.txt
├── models/
│   └── best_model.pt
└── restored_test_outputs/
    └── 000000.npy ...
```

## 1. Environment Setup

Python 3.11 or newer is recommended. A CUDA-capable GPU is recommended for
training and fast inference; CPU inference is also supported.

```bash
git clone [REPLACE_WITH_PUBLIC_REPOSITORY_URL]
cd [REPLACE_WITH_REPOSITORY_DIRECTORY]

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The scripts automatically use CUDA when available and otherwise fall back to
CPU. To reproduce the reported latency, use an NVIDIA GPU and `--device cuda`.

## 2. Run Inference

The evaluation script accepts the two required positional arguments:

1. directory containing degraded `.npy` inputs;
2. directory in which restored `.npy` outputs will be written.

```bash
python evaluate.py /path/to/NoisyLR /path/to/restored_outputs
```

Explicit CUDA example:

```bash
python evaluate.py /path/to/NoisyLR /path/to/restored_outputs \
  --checkpoint models/best_model.pt \
  --device cuda \
  --batch-size 16
```

CPU example:

```bash
python evaluate.py /path/to/NoisyLR /path/to/restored_outputs \
  --device cpu \
  --batch-size 4
```

No source edits or per-image configuration are required. For every input named
`000123.npy`, the script writes `restored_outputs/000123.npy`.

### Input Contract

- File format: NumPy `.npy`
- Shape: `128x128`
- Content: one numeric grayscale image
- Values: finite; values outside `[0, 1]` are allowed and are not clipped

### Output Contract

- File format: NumPy `.npy`
- Shape: `256x256`
- Data type: `float32`
- Range: `[0, 1]`
- Filename: identical to the corresponding input filename

Show all inference options:

```bash
python evaluate.py --help
```

## 3. Reproduce Training

Arrange the provided paired training data as follows:

```text
/path/to/train/
├── NoisyLR/
│   ├── 000000.npy
│   └── ...
└── GT/
    ├── 000000.npy
    └── ...
```

Matching noisy and ground-truth arrays must have identical filenames. Run:

```bash
python train.py /path/to/train \
  --checkpoint models/reproduced_model.pt \
  --device cuda
```

The default command reproduces the selected configuration:

- fixed split seed: 2026;
- validation images: 320 (10%);
- model seed: 1337;
- batch size: 16;
- training budget: 300 seconds;
- AdamW peak learning rate: `4e-4`;
- weight decay: `1e-4`;
- 5% linear warmup followed by cosine decay;
- Charbonnier pixel loss + gradient loss + SSIM loss;
- paired 90-degree rotations and horizontal flips;
- BF16 mixed precision on CUDA.

Training is time-budgeted, so the exact number of optimizer steps depends on
hardware and storage throughput. The submitted H200 run completed 33,711 steps
and saw 539,376 augmented examples in 300 seconds.

Show all training options:

```bash
python train.py --help
```

## Model Design

The model uses:

- a 32-channel input convolution;
- two residual encoder levels;
- four residual blocks in the bottleneck;
- PixelShuffle decoder stages with U-Net skip connections;
- a final learned 2x correction;
- a bicubic 2x residual path carrying coarse image structure.

The degraded input is deliberately not clipped. Speckle noise can create valid
excursions outside the clean-image range, and clipping would discard useful
signal. Only the final saved restoration is clipped to `[0, 1]`.

## Submitted Artifacts

- `models/best_model.pt`: final `lr_4e4` checkpoint.
- `restored_test_outputs/`: restored results for all 400 provided test arrays.
- `evaluate.py`: standalone evaluation entry point.
- `train.py`: complete training and validation implementation.
- `requirements.txt`: pinned clean reproducibility environment.

## Troubleshooting

### CUDA requested but unavailable

Install a PyTorch build compatible with the installed NVIDIA driver, then
confirm `torch.cuda.is_available()` returns `True`. Alternatively, run with
`--device cpu`.

### Out-of-memory error

Reduce `--batch-size`, for example:

```bash
python evaluate.py /path/to/NoisyLR /path/to/outputs --batch-size 4
```

### Unexpected input shape

The submitted checkpoint was trained for `128x128 -> 256x256` restoration.
The evaluator intentionally rejects other shapes instead of silently resizing
them.

## License and Data

The trained weights and code are supplied for hackathon evaluation. The
competition dataset is not redistributed in this repository; obtain it through
the official hackathon channel and follow its usage terms.

# KLA-hackathon-submission
