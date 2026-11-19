# Reconstruction verification

Checked on 2026-11-18 with Python 3.12, PyTorch 2.10.0+cu129 and NVIDIA A800-SXM4-80GB.

| Check | Result |
| --- | --- |
| Export selected K7/K23 50k codecs | 842 state tensors per model; every tensor exactly equals the selected original value, without precision conversion |
| Anonymous public HF download | Both complete checkpoints downloaded with `token=False`; downloaded files match the upload exports byte for byte |
| K7 reconstruction from the downloaded checkpoint | Exit 0; 256×256 RGBA PNG |
| K23 reconstruction from the downloaded checkpoint | Exit 0; 1024×1024 RGBA PNG |
| Complete strict state loading | No missing or unexpected keys for either codec |
| External model dependencies during inference | None; only the selected complete codec file was opened as a model input |
| K7 comparison with the original checkpoint reference | Same procedural input; output pixels and PNG bytes are identical |

The download command succeeded on the networked host. The GPU worker had no route to Hugging Face, so its direct anonymous download attempt failed with `Network is unreachable`. The exact files downloaded on the networked host were then transferred to that worker. An independent agent ran both README reconstruction commands with Hugging Face and Transformers offline modes enabled, without any separate DINOv3 checkpoint, RGB initializer or model links. All twelve saved PNG views were readable. Detailed local logs preserve the failed network attempt and the successful offline GPU inference as separate stages.

The existing-environment editable install also passed. This verifies reuse of the recorded dependencies; a fresh network-based installation of all dependencies is not claimed as tested.

## One-command training initialization and launch

The public training bundle contains the original K7/K23 RGB decoder initializers, frozen DINOv3-L encoder, AlexNet trunk and LPIPS-Alex cache. Every initializer decoder tensor exactly matches the original RGB starting point, and all five anonymous model downloads match the staged upload bundle byte for byte. Final RGBA checkpoints are not used to initialize these tests.

Both configurations were exercised with `scripts.launch_train` on eight A800 GPUs and global batch eight, starting from an isolated model directory containing only the corresponding bundle. K7 was checked by an independent agent following the README commands. Each run used eight real training assets and two validation assets, completed two optimizer updates, saved a checkpoint, and resumed to update three.

| Check | K7 | K23 |
| --- | --- | --- |
| Original RGB initialization | Passed; 456 decoder tensors match; new readout output starts at zero | Passed; selected native 1024 RGB initializer |
| Optimizer states after resume | 475 | 475 |
| Saved rank RNG states | 8 | 8 |
| Finite nonzero training gradients and changed parameters | Passed | Passed |
| LPIPS training supervision | Passed using downloaded AlexNet plus calibration packaged with the LPIPS dependency | Passed |
| Default online PSNR/SSIM/LPIPS validation | Passed using the staged metric caches | Passed |
| Export the trained step-three codec and run reconstruction | 256×256 RGBA, strict loading | 1024×1024 RGBA, strict loading |

The GPU tests ran without Internet access. Training bundle files were staged to the isolated directories; the complete published bundle was also anonymously downloaded on the connected host and byte-compared. The small LPIPS calibration file distributed inside the installed LPIPS package is a standard dependency asset, while the AlexNet trunk and PyIQA calibration use the downloaded training cache. No diffusion prior or generation adapter was present or accessed.

The tested environment also passed existing-dependency editable installation, and the pinned PyIQA 0.1.16 release was confirmed on the official PyPI index. A fresh network installation of all dependencies and full 50k convergence training are not claimed as tested.

## ImageNet class-conditioned RGBA generation

The restored public pipeline uses the selected K7 RGB DiT EMA, image-only adapter EMA and matching normalization. All exported model tensors exactly match their original sources. The reused RGB decoder initializer was also compared against the native RGB decoder: all 456 state tensors match.

The class-generation CLI ran on one A800 with classes 985/849 and seeds 20260926/20261118. All four 256×256 RGBA outputs are pixel-identical and PNG-byte-identical to the preserved original pipeline references. The existing reconstruction entrypoint also completed successfully with the restored sampling runtime. The anonymous download script fetched all new generation files and shared dependencies; downloaded model files exactly match the upload exports.

The pipeline accepts only ImageNet class IDs and random seeds. It samples RGB first, then predicts the RGBA decoder input. It does not access target images, alpha labels, a matting teacher or a text-conditioned model. These are engineering and qualitative sample checks, not a new ImageNet benchmark or a success-rate estimate.

An independent agent installed the existing-dependency editable package and ran the README generation command unchanged in an isolated GPU directory using only the five public model files. Both 256×256 RGBA outputs match the primary test pixels and PNG bytes exactly. The executed source uses strict state loading for the prior/adapter/codec/RGB decoder, whose saved state counts are 572/217/842/456. GPU-side execution was offline; the connected parent had separately verified anonymous downloads and transferred those exact files. No text model, dataset, target image or teacher was used.
