# AlphaRAE

**Joint Color and Opacity Modeling with Representation Autoencoders**

RGBA reconstruction with a frozen DINOv3-L encoder and a trained joint color-opacity decoder. This release provides **K7 at 256×256** and **K23 at 1024×1024**, reconstruction inference, and codec training.

## Installation

Use Linux, Python 3.12 and an NVIDIA GPU. The verified environment uses PyTorch 2.10.0 with CUDA 12.9 and torchvision 0.25.0.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -e .
```

For an environment with these dependencies already installed:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

Run the commands below from this repository root.

## Download weights

The public reconstruction checkpoints are hosted at [TonyZhan/AlphaRAE-Reconstruction](https://huggingface.co/TonyZhan/AlphaRAE-Reconstruction). Download both models directly into `weights/` with one command:

```bash
python -m scripts.download_weights
```

No Hugging Face account or token is required. To download one configuration:

```bash
python -m scripts.download_weights --codec k7
python -m scripts.download_weights --codec k23
```

| File | Resolution | Selected checkpoint | Download size |
| --- | --- | --- | --- |
| `weights/codec_k7.pt` | 256×256 | 50,000 updates | 2.88 GB |
| `weights/codec_k23.pt` | 1024×1024 | 50,000 updates | 2.89 GB |

On compute nodes without Internet access, run the download command on a connected machine and copy the resulting `weights/` directory to the compute node.

Each file includes the frozen encoder and trained reconstruction modules. **No separate DINOv3 model, RGB initializer, or normalization statistics are needed for inference.** The download command also saves the component licenses and model manifest. See [weights/README.md](weights/README.md).

## Reconstruct RGBA

```bash
python -m scripts.infer --codec k7 --input examples/rgba.png --output outputs/reconstruction_k7
python -m scripts.infer --codec k23 --input examples/rgba.png --output outputs/reconstruction_k23
```

Replace `examples/rgba.png` with your straight-alpha, 8-bit RGBA PNG. The included example is a procedural test asset. Images are resized with premultiplied alpha and padded to the selected square resolution. The loader uses linear RGB internally; saved RGBA PNGs use sRGB and straight alpha. Output folders contain reconstruction RGBA, foreground RGB, alpha, black/white/checkerboard views, and `report.json`. Use `--device cpu` for CPU inference or `--weights-dir /path/to/weights` for a custom model directory.

## Data preparation

Keep related assets in the same split and convert tagged images to untagged sRGB before preparing the manifest:

```bash
python -m scripts.prepare_data --train /data/train --val /data/val --test /data/test --output data/rgba_manifest.jsonl
```

See [data/README.md](data/README.md) for the RGBA and RGB replay manifest formats. The original reconstruction study used 7,779 train, 422 validation and 434 local test assets. Datasets and private records are not included in this repository.

## Codec training

The original eight-GPU configuration uses batch 1 per rank, global batch 8, full decoder training and 50,000 optimizer updates. Training additionally needs the matching RGB decoder initializer, a DINOv3-L checkpoint and the data paths specified in the configuration. These training assets are separate from the self-contained inference downloads.

```bash
python -m pip install -e '.[evaluation]'
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.train_codec --config configs/codec_k7.yaml --output outputs/codec_k7
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.train_codec --config configs/codec_k23.yaml --output outputs/codec_k23
```

For another number of ranks, adjust `batch_size`, `effective_batch_size` and `planned_world_size` together. Resume with the same configuration and world size:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.train_codec --config configs/codec_k7.yaml --output outputs/codec_k7 --resume outputs/codec_k7/last.pt
```

Before a full job, the startup check performs two real updates, saves, resumes to update three, and checks gradients and parameter changes:

```bash
python -m scripts.smoke_codec --config configs/codec_k7.yaml --output outputs/smoke_k7
python -m scripts.smoke_codec --config configs/codec_k23.yaml --output outputs/smoke_k23
```

Export a trained checkpoint to the portable inference format:

```bash
python -m scripts.export_reconstruction --checkpoint outputs/codec_k7/last.pt --output weights/codec_k7.pt
```

## Verification and attribution

[docs/verification.md](docs/verification.md) records the checks and their scope. The implementation builds on [RAE](https://github.com/bytetriper/RAE), [RAEv2](https://github.com/nanovisionx/RAEv2) and [DINOv3](https://github.com/facebookresearch/dinov3). Runtime source and original licenses are retained in `third_party/`; see [THIRD_PARTY.md](THIRD_PARTY.md). RAEv2 materials use CC BY-NC 4.0 and DINOv3 materials retain their own license. The root MIT notice does not override upstream code, model or dataset licenses.
