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

The original codec training implementation was exercised separately at both 256 and 1024 pixels: two real updates, checkpoint saving, and resume to update three. Nonzero gradients, 475 optimizer states and changed reconstruction parameters were observed. That startup check retains the reconstruction/fidelity/LPIPS losses, while disabling optional online JPEG metric evaluation. The full 50k training budget was not rerun for this release.
