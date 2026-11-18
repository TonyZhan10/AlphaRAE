# Reconstruction weights

Download the complete reconstruction codecs from [TonyZhan/AlphaRAE-Reconstruction](https://huggingface.co/TonyZhan/AlphaRAE-Reconstruction):

```bash
python -m scripts.download_weights
```

| File | Original selected artifact | Resolution |
| --- | --- | --- |
| `codec_k7.pt` | `raev2_k7_rgba256_full/step_0050000.pt`, exported model tensors | 256×256 |
| `codec_k23.pt` | `rgba_sft50k_full/step_0050000.pt`, exported model tensors | 1024×1024 |

Both portable files include the frozen DINOv3-L encoder, trained shared decoder and reconstruction detail/readout modules. They contain no optimizer or RNG state. All 842 state tensors per model were checked against the selected original checkpoint without dtype conversion. The release loader constructs the architecture without loading external pretrained files, then loads the complete state strictly with `weights_only=True`.

The downloader places component licenses and `reconstruction_manifest.json` alongside the models. Public inference downloads need no token. Training from the original RGB starting point additionally uses `training/rgb_k7_init.pt` or `training/rgb_k23_init.pt`, `training/dinov3/` and perceptual-network caches under `training/torch/`; these are prepared by the training downloader and are not required to reconstruct an RGBA image.

## Training initialization

```bash
python -m scripts.download_training_weights
```

This separately downloads both original RGB decoder initializers and the shared DINOv3/AlexNet/LPIPS files under `training/`. Use `--codec k7` or `--codec k23` to fetch one initializer. The initializers are the original RGB starting points used by the reconstruction experiments, rather than the final RGBA models. All decoder tensors were checked against those original inputs. Additional component licenses and the training manifest are downloaded with the files.
