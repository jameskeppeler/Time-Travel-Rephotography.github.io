# Windows Setup

This document covers the verified Windows setup, asset bootstrapping, and wrapper workflow for this fork.

## Windows CUDA 11.8 workflow

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

- `rephoto_cuda11_working.yml` — original working export from the reference machine
- `rephoto_cuda11_portable.yml` — recommended for new installs on other machines

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
### Optional wrapper config file

If `rephoto_wrapper.config.json` is present in the repo root, the wrappers will use it to supply default values for common paths and environment names.

This lets you avoid repeating the same parameters on every run. Explicit command-line arguments still take precedence over the config file.

Example config file:

```json
{
  "GFPGANRoot": ".\\deps\\GFPGAN",
  "GFPGANEnvName": "gfpgan_py38",
  "FaceCropEnvName": "facecrop_py310",
  "FaceCropCommand": "face-crop-plus",
  "RephotoEnvName": "rephoto_cuda11",
  "EncoderCkptPath": ".\\checkpoint\\encoder\\checkpoint_g.pt",
  "ProjectorScriptPath": ".\\projector.py",
  "PreprocessRoot": ".\\preprocess",
  "ResultsRoot": ".\\results"
}

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