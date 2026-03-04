# Time-Travel Rephotography
<a href="https://arxiv.org/abs/2012.12261"><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KZXGkHVhvz2X3ljaCQANC1bDr7OrzDpg?usp=sharing)

### [Project Website](https://time-travel-rephotography.github.io/)

<p align='center'>
<img src="time-travel-rephotography.gif" width='100%'/>
</p>

Many historical people were only ever captured by old, faded, black-and-white photos distorted by the limitations of early cameras and the passage of time. This project simulates traveling back in time with a modern camera to rephotograph historical subjects. Unlike conventional restoration filters that apply denoising, colorization, and super-resolution as separate steps, this method projects old photos into the space of modern high-resolution portraits using StyleGAN2, combining these effects in a unified pipeline.

A core challenge is preserving identity and pose while discarding artifacts common in low-quality antique photographs. The original paper demonstrates strong qualitative improvements over restoration baselines on a range of historical portraits.

**Time-Travel Rephotography**  
Xuan Luo, Xuaner Zhang, Paul Yoo, Ricardo Martin-Brualla, Jason Lawrence, and Steven M. Seitz  
SIGGRAPH Asia 2021.

---

## What this repository now includes

This repository now contains **two related workflows**:

1. **Original upstream workflow** for the research projector.
2. **A Windows-focused, developer-friendly wrapper workflow** that adds:
   - automatic face cropping from raw images,
   - optional GFPGAN restoration,
   - conservative blend-back of restored faces,
   - a configurable projector runner for one or many cropped faces.

This fork has also been refactored to be **more portable across machines** by making key paths and environment names configurable.

---

## Windows CUDA 11.8 workflow (recommended for this fork)

This repository has been verified in a Windows environment for the projector workflow after significant compatibility fixes.

### Known-good references

- Main repo tag: `rephoto_cuda11_working_2026-02-26_bootstrap_v2`
- Main repo commit: `fa95639`
- Prior restore tag: `rephoto_cuda11_working_2026-02-25_postrestore`
- Submodule tag: `encoder4editing_cuda11_working_2026-02-25`
- Submodule commit: `9520744f95c9109c3cfcd5ca1f5f0dc6da70541f`

A clean restore clone was tested successfully using this baseline.

### Verified working environment

- Environment name: `rephoto_cuda11`
- Python: `3.8`
- PyTorch: `2.4.1`
- CUDA runtime: `11.8`
- GPU verified: `NVIDIA GeForce RTX 3060 Laptop GPU`

Additional required packages in the working environment:

- `opencv`
- `tensorboard`
- `matplotlib`
- `tqdm`
- `scikit-image`

Environment files included in this repo:

- `rephoto_cuda11_working.yml` — original working export
- `rephoto_cuda11_portable.yml` — portable copy with the machine-specific `prefix:` removed

---

## Required local assets

A fresh clone may still need these local assets before the projector can run:

1. Test input image:
   - `dataset\Abraham Lincoln_01.png`
2. Main checkpoints:
   - `checkpoint\*`
3. Face parsing checkpoint:
   - `third_party\face_parsing\res\cp\79999_iter.pth`

---

## Bootstrapping local assets

The repository includes `bootstrap_local_assets.ps1` to restore required local assets.

### Copy from an existing populated repo

```powershell
.\bootstrap_local_assets.ps1 -Mode copy -SourceRepo "C:\Path\To\Populated\Repo"
```

### Download assets directly (no `SourceRepo` required)

Recommended (Hugging Face mirror):

```powershell
.\bootstrap_local_assets.ps1 -Mode download -DownloadProvider hf
```

Alternative (official Google Drive IDs; may hit quota):

```powershell
.\bootstrap_local_assets.ps1 -Mode download -DownloadProvider gdrive
```

### Auto mode

Default behavior:

- if `-SourceRepo` is provided, the script copies from that repo,
- otherwise it downloads the assets.

```powershell
.\bootstrap_local_assets.ps1
```

---

## Verified restore procedure

A clean restore test was successfully performed from a parent folder.

1. Clone the repository with submodules:

   ```powershell
   git clone --recurse-submodules <repo-url>
   ```

2. Check out the verified tag:

   ```powershell
   git checkout rephoto_cuda11_working_2026-02-26_bootstrap_v2
   ```

3. Confirm the `encoder4editing` submodule resolves to:

   - `9520744f95c9109c3cfcd5ca1f5f0dc6da70541f`

4. Restore local assets with `bootstrap_local_assets.ps1`.

5. Run the projector test command.

---

## Verified projector commands

### Stage-1 run (32x32 only)

```powershell
python projector.py "dataset\Abraham Lincoln_01.png" `
  --encoder_ckpt "checkpoint\encoder\checkpoint_g.pt" `
  --color_transfer 0 `
  --eye 0 `
  --lr 0.001 `
  --noise_regularize 0 `
  --camera_lr 0 `
  --wplus_step 250 `
  --results_dir "results/projector_restore_test_v2"
```

### Full run (32x32 + 64x64)

```powershell
python projector.py "dataset\Abraham Lincoln_01.png" `
  --encoder_ckpt "checkpoint\encoder\checkpoint_g.pt" `
  --color_transfer 0 `
  --eye 0 `
  --lr 0.001 `
  --noise_regularize 0 `
  --camera_lr 0 `
  --results_dir "results/projector_restore_test_full_v2"
```

### What a successful projector run confirms

A successful run confirms the full intended projector protocol is functioning:

1. initialize latent code (encoder / e4e path),
2. generate the initial sibling image,
3. run face parsing and skin-mask generation,
4. run histogram matching,
5. save the initial image and latent/noise state,
6. optimize through the configured W+ stages,
7. save the final image, final latent/noise `.pt`, and random-noise variant.

---

## Integrated face-crop + GFPGAN wrapper workflow

This fork includes a Windows wrapper workflow for running the projector directly from raw face photos.

Wrapper scripts:

- `run_rephoto_with_facecrop.ps1`
- `run_rephoto_with_facecrop_batch.ps1`

These wrappers can:

1. detect and crop faces with `face-crop-plus`,
2. optionally enhance cropped faces with GFPGAN,
3. blend the GFPGAN result conservatively back into the original crop,
4. run the Time-Travel Rephotography projector on the final prepared crop(s).

### Recommended preprocessing settings

Current tested settings:

- face crop strategy: `all`
- face factor: `0.65`
- detection threshold: `0.9`

Current conservative GFPGAN settings:

- `-UseGFPGAN`
- `-GFPGANVersion 1.3`
- `-GFPGANBlend 0.35`

In practice, the best visual balance so far has been:

- run GFPGAN,
- then blend only **35%** of the restored result back into the original cropped face.

This preserves more original facial detail while reducing obvious restoration artifacts.

### Preset rules

`-Preset` accepts:

- `test` (maps to `wplus_step 250 750`)
- `1500`
- any multiple of `1000` from `1000` through `100000`

Examples:

- `1000`
- `1500`
- `3000`
- `6000`
- `18000`
- `50000`
- `100000`

Internal mapping:

- `test` -> `wplus_step 250 750`
- numeric preset `N` -> `wplus_step 250 N`

### Practical quality tiers

Recommended presets for day-to-day use:

- `1500` — best for generating a Photoshop color-overlay base from the original image
- `3000` — best balance of quality and runtime
- `6000` — best standard high-quality preset when longer runtime is acceptable

A tested `18000` run produced the best visual result so far, but was much slower on the reference GPU.

### Runtime trend on the reference GPU

Approximate timings observed on an **NVIDIA GeForce RTX 3060 Laptop GPU**:

- `750` -> about **10 min**
- `1500` -> about **20 min**
- `3000` -> about **38 min**
- `6000` -> about **2 h 16 min**
- `18000` -> about **7 h 47 min**

Above `6000`, estimates are extrapolated and should be treated as approximate. Actual runtime depends on GPU model, VRAM, thermals, background load, and the installed CUDA / PyTorch stack.

### Portable wrapper usage

```powershell
.\run_rephoto_with_facecrop.ps1 `
  -InputImage ".\dataset\unk.jpg" `
  -Preset 3000 `
  -Strategy all `
  -FaceFactor 0.65 `
  -DetThreshold 0.9 `
  -UseGFPGAN `
  -GFPGANVersion 1.3 `
  -GFPGANBlend 0.35 `
  -GFPGANRoot ".\deps\GFPGAN" `
  -FaceCropEnvName "facecrop_py310" `
  -FaceCropCommand "face-crop-plus" `
  -RephotoEnvName "rephoto_cuda11"
```

### Example single-image wrapper usage

```powershell
.\run_rephoto_with_facecrop.ps1 `
  -InputImage ".\dataset\unk.jpg" `
  -Preset 3000 `
  -Strategy all `
  -FaceFactor 0.65 `
  -DetThreshold 0.9 `
  -CropIndex 0 `
  -UseGFPGAN `
  -GFPGANVersion 1.3 `
  -GFPGANBlend 0.35
```

### Batch wrapper usage

```powershell
.\run_rephoto_with_facecrop_batch.ps1 `
  -InputDir ".\dataset\batch" `
  -Preset 3000 `
  -Strategy all `
  -FaceFactor 0.65 `
  -DetThreshold 0.9 `
  -UseGFPGAN `
  -GFPGANVersion 1.3 `
  -GFPGANBlend 0.35
```

### Configurable wrapper parameters

The wrappers now support configurable values for better portability:

- `GFPGANRoot`
- `GFPGANEnvName`
- `FaceCropEnvName`
- `FaceCropCommand`
- `RephotoEnvName`
- `EncoderCkptPath`
- `ProjectorScriptPath`
- `PreprocessRoot`
- `ResultsRoot`

---

## Original upstream workflow (legacy / reference)

The original upstream workflow remains useful as a reference and for non-Windows setups.

### Demo

A Google Colab demo is available for trying the method on the Abraham Lincoln example image or on your own photos using cloud GPUs.

### Original prerequisites

- Pull third-party packages:

  ```bash
  git submodule update --init --recursive
  ```

- Install Python packages:

  ```bash
  conda create --name rephotography python=3.8.5
  conda activate rephotography
  conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
  pip install -r requirements.txt
  ```

### Original quick start

```bash
./scripts/download_checkpoints.sh
./scripts/run.sh b "dataset/Abraham Lincoln_01.png" 0.75
```

Inspect optimization progress with:

```bash
tensorboard --logdir "log/Abraham Lincoln_01"
```

### Run on your own image

- Crop and align head regions:

  ```bash
  python -m tools.data.align_images <input_raw_image_dir> <aligned_image_dir>
  ```

- Run the projector:

  ```bash
  ./scripts/run.sh <spectral_sensitivity> <input_image_path> <blur_radius>
  ```

`spectral_sensitivity` can be:

- `b` — blue-sensitive
- `gb` — orthochromatic
- `g` — panchromatic

A rough guideline:

- before 1873: use `b`
- 1873–1906: manually choose between `b` and `gb`
- after 1906: any of the three may be relevant depending on the photo

`blur_radius` is the estimated Gaussian blur radius in pixels if the input is resized to 1024×1024.

---

## Historical Wiki Face Dataset

| Path | Size | Description |
| --- | --- | --- |
| Historical Wiki Face Dataset.zip | 148 MB | Images |
| spectral_sensitivity.json | 6 KB | Spectral sensitivity (`b`, `gb`, or `g`) |
| blur_radius.json | 6 KB | Blur radius in pixels |

The JSON files map input names to the corresponding spectral sensitivity or blur radius.

Due to copyright constraints, the zip file excludes the Mao Zedong image used in the original study. That image must be obtained separately and cropped manually if needed.

---

## Citation

If you find this code useful, please consider citing the original paper:

```bibtex
@article{Luo-Rephotography-2021,
  author    = {Luo, Xuan and Zhang, Xuaner and Yoo, Paul and Martin-Brualla, Ricardo and Lawrence, Jason and Seitz, Steven M.},
  title     = {Time-Travel Rephotography},
  journal   = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH Asia 2021)},
  publisher = {ACM New York, NY, USA},
  volume    = {40},
  number    = {6},
  articleno = {213},
  doi       = {https://doi.org/10.1145/3478513.3480485},
  year      = {2021},
  month     = {12}
}
```

---

## License

This work is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Code for the StyleGAN2 model comes from `rosinality/stylegan2-pytorch`.

---

## Acknowledgments

We thank Nick Brandreth for capturing the dry plate photos. We also thank Bo Zhang, Qingnan Fan, Roy Or-El, Aleksander Holynski, and Keunhong Park for helpful discussions, and Xiaojie Feng for contributions to the Colab demo.
