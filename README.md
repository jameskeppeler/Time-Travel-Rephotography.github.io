# [SIGGRAPH Asia 2021] Time-Travel Rephotography
<a href="https://arxiv.org/abs/2012.12261"><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KZXGkHVhvz2X3ljaCQANC1bDr7OrzDpg?usp=sharing)
### [[Project Website](https://time-travel-rephotography.github.io/)]
<p align='center'>
<img src="time-travel-rephotography.gif" width='100%'/>
</p>

Many historical people were only ever captured by old, faded, black and white photos, that are distorted due to the limitations of early cameras and the passage of time. This paper simulates traveling back in time with a modern camera to rephotograph famous subjects. Unlike conventional image restoration filters which apply independent operations like denoising, colorization, and superresolution, we leverage the StyleGAN2 framework to project old photos into the space of modern high-resolution photos, achieving all of these effects in a unified framework. A unique challenge with this approach is retaining the identity and pose of the subject in the original photo, while discarding the many artifacts frequently seen in low-quality antique photos. Our comparisons to current state-of-the-art restoration filters show significant improvements and compelling results for a variety of important historical people. 
<br/>

**Time-Travel Rephotography**
<br/>
[Xuan Luo](https://roxanneluo.github.io),
[Xuaner Zhang](https://people.eecs.berkeley.edu/~cecilia77/),
[Paul Yoo](https://www.linkedin.com/in/paul-yoo-768a3715b),
[Ricardo Martin-Brualla](http://www.ricardomartinbrualla.com/),
[Jason Lawrence](http://jasonlawrence.info/), and 
[Steven M. Seitz](https://homes.cs.washington.edu/~seitz/)
<br/>
In SIGGRAPH Asia 2021.

## Demo
We provide an easy-to-get-started demo using Google Colab!
The Colab will allow you to try our method on the sample Abraham Lincoln photo or **your own photos** using Cloud GPUs on Google Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KZXGkHVhvz2X3ljaCQANC1bDr7OrzDpg?usp=sharing)

Or you can run our method on your own machine following the instructions below.
 
## Prerequisite
- Pull third-party packages.
  ```
  git submodule update --init --recursive
  ```
- Install python packages.
  ```
  conda create --name rephotography python=3.8.5
  conda activate rephotography
  conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
  pip install -r requirements.txt
  ```

## Quick Start
Run our method on the example photo of Abraham Lincoln.
- Download models:
  ```
  ./scripts/download_checkpoints.sh
  ```
- Run:
  ```
  ./scripts/run.sh b "dataset/Abraham Lincoln_01.png" 0.75 
  ```
- You can inspect the optimization process by  
  ```
  tensorboard --logdir "log/Abraham Lincoln_01"
  ```
- You can find your results as below.
  ```
  results/
    Abraham Lincoln_01/       # intermediate outputs for histogram matching and face parsing
    Abraham Lincoln_01_b.png  # the input after matching the histogram of the sibling image
    Abraham Lincoln_01-b-G0.75-init(10,18)-s256-vgg1-vggface0.3-eye0.1-color1.0e+10-cx0.1(relu3_4,relu2_2,relu1_2)-NR5.0e+04-lr0.1_0.01-c32-wp(250,750)-init.png        # the sibling image
    Abraham Lincoln_01-b-G0.75-init(10,18)-s256-vgg1-vggface0.3-eye0.1-color1.0e+10-cx0.1(relu3_4,relu2_2,relu1_2)-NR5.0e+04-lr0.1_0.01-c32-wp(250,750)-init.pt         # the sibing latent codes and initialized noise maps
    Abraham Lincoln_01-b-G0.75-init(10,18)-s256-vgg1-vggface0.3-eye0.1-color1.0e+10-cx0.1(relu3_4,relu2_2,relu1_2)-NR5.0e+04-lr0.1_0.01-c32-wp(250,750).png             # the output result
    Abraham Lincoln_01-b-G0.75-init(10,18)-s256-vgg1-vggface0.3-eye0.1-color1.0e+10-cx0.1(relu3_4,relu2_2,relu1_2)-NR5.0e+04-lr0.1_0.01-c32-wp(250,750).pt              # the final optimized latent codes and noise maps
    Abraham Lincoln_01-b-G0.75-init(10,18)-s256-vgg1-vggface0.3-eye0.1-color1.0e+10-cx0.1(relu3_4,relu2_2,relu1_2)-NR5.0e+04-lr0.1_0.01-c32-wp(250,750)-rand.png        # the result with the final latent codes but random noise maps

  ```

## Run on Your Own Image
- Crop and align the head regions of your images:
  ```
  python -m tools.data.align_images <input_raw_image_dir> <aligned_image_dir>
  ```
- Run:
  ```
  ./scripts/run.sh <spectral_sensitivity> <input_image_path> <blur_radius>
  ```
  The `spectral_sensitivity` can be `b` (blue-sensitive), `gb` (orthochromatic), or `g` (panchromatic). You can roughly estimate the `spectral_sensitivity` of your photo as follows. Use the *blue-sensitive* model for photos before 1873, manually select between blue-sensitive and *orthochromatic* for images from 1873 to 1906 and among all models for photos taken afterwards.

  The `blur_radius` is the estimated gaussian blur radius in pixels if the input photot is resized to 1024x1024.
  
## Historical Wiki Face Dataset
| Path      | Size | Description |
|----------- | ----------- | ----------- |
| [Historical Wiki Face Dataset.zip](https://drive.google.com/open?id=1mgC2U7quhKSz_lTL97M-0cPrIILTiUCE&authuser=xuanluo%40cs.washington.edu&usp=drive_fs)| 148 MB | Images|
| [spectral_sensitivity.json](https://drive.google.com/open?id=1n3Bqd8G0g-wNpshlgoZiOMXxLlOycAXr&authuser=xuanluo%40cs.washington.edu&usp=drive_fs)| 6 KB | Spectral sensitivity (`b`, `gb`, or `g`). |
| [blur_radius.json](https://drive.google.com/open?id=1n4vUsbQo2BcxtKVMGfD1wFHaINzEmAVP&authuser=xuanluo%40cs.washington.edu&usp=drive_fs)| 6 KB | Blur radius in pixels| 

The `json`s are dictionares that map input names to the corresponding spectral sensitivity or blur radius.
Due to copyright constraints, `Historical Wiki Face Dataset.zip` contains all images in the *Historical Wiki Face Dataset* that were used in our user study except the photo of [Mao Zedong](https://en.wikipedia.org/wiki/File:Mao_Zedong_in_1959_%28cropped%29.jpg). You can download it separately and crop it as [above](#run-on-your-own-image). 

## Citation
If you find our code useful, please consider citing our paper:
```
@article{Luo-Rephotography-2021,
  author    = {Luo, Xuan and Zhang, Xuaner and Yoo, Paul and Martin-Brualla, Ricardo and Lawrence, Jason and Seitz, Steven M.},
  title     = {Time-Travel Rephotography},
  journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH Asia 2021)},
  publisher = {ACM New York, NY, USA},
  volume = {40},
  number = {6},
  articleno = {213},
  doi = {https://doi.org/10.1145/3478513.3480485},
  year = {2021},
  month = {12}
}
```

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

Codes for the StyleGAN2 model come from [https://github.com/rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).

## Acknowledgments
We thank [Nick Brandreth](https://www.nickbrandreth.com/) for capturing the dry plate photos. We thank Bo Zhang, Qingnan Fan, Roy Or-El, Aleksander Holynski and Keunhong Park for insightful advice. We thank [Xiaojie Feng](https://github.com/fengxiaojie-coder) for his contributions on the colab demo.
## Windows CUDA 11.8 troubleshooting and verified working workflow

This repository was successfully brought to a working state on Windows for the projector workflow after extensive troubleshooting. This section records the final known-good configuration, the code changes required, the local assets required, and the restore procedure that was verified in a fresh clone.

### Known-good references

- Main repo tag: `rephoto_cuda11_working_2026-02-26_bootstrap_v2`
- Main repo commit: `fa95639`
- Prior restore tag: `rephoto_cuda11_working_2026-02-25_postrestore`
- Submodule tag: `encoder4editing_cuda11_working_2026-02-25`
- Submodule commit: `9520744f95c9109c3cfcd5ca1f5f0dc6da70541f`

A clean restore clone was tested and the projector workflow ran successfully after restoring required local assets.

### What failed originally

The original Windows setup failed for several separate reasons:

1. The old environment (`torch 1.4.0` + `CUDA 10.1`) eventually saw the GPU after driver/device troubleshooting, but it produced corrupted or blocky saved outputs on modern Ampere hardware.
2. Both the main repo and the `encoder4editing` submodule rely on StyleGAN operator paths that normally JIT-compile custom CUDA/C++ extensions. That is fragile on Windows and was not reliable in this setup.
3. A fresh CUDA11 environment initially did not contain all required Python packages for the full projector pipeline.
4. Newer `torchvision` changed the `make_grid()` API from `range=` to `value_range=`, which caused a runtime crash during logging.
5. A fresh clone does not include all required local assets (test image, checkpoints, face parsing weights), so the pipeline cannot run until those are restored.

### Code changes made

#### `utils/torch_helpers.py`

The image conversion path was changed so tensors are converted on CPU before casting to `uint8`.

Reason:
- This avoids corrupted/blocky saved outputs seen with the older GPU-side conversion path.

Functional behavior:
- detach tensor
- clamp safely
- scale from `[-1, 1]` to `[0, 255]`
- move to CPU
- cast to `uint8`
- return contiguous HWC NumPy array

#### `projector.py`

The OpenCV save path was changed so the RGB-to-BGR flip is followed by `.copy()`.

Change:
- `img[..., ::-1]` -> `img[..., ::-1].copy()`

Reason:
- This ensures the array passed to `cv2.imwrite()` is contiguous and avoids negative-stride save issues.

#### `op/upfirdn2d.py`

The top-level `upfirdn2d` path was patched so the delegated e4e op is used only when:
- the delegated implementation exists, and
- the input tensor is actually CUDA

Otherwise it falls back to the pure PyTorch implementation.

Reason:
- This prevents invalid delegation on CPU and preserves a safe fallback path.

#### `utils/optimize.py`

`torchvision.utils.make_grid()` calls were updated for newer `torchvision`.

Change:
- `range=(-1, 1)` -> `value_range=(-1, 1)`

Reason:
- Without this, projector execution crashed during visual logging.

#### `models/encoder4editing/models/stylegan2/op/upfirdn2d.py`

The submodule `upfirdn2d` operator was patched to avoid `torch.utils.cpp_extension.load()` JIT compilation on Windows.

Reason:
- Instead of compiling a custom extension at import time, this file now routes to the repository's pure PyTorch fallback implementation.
- This is the key Windows/CUDA11 compatibility fix for the submodule operator path.

#### Other tracked files in the known-good state

These files were also part of the known-good working tree during troubleshooting and should remain pinned to the verified repo/submodule revisions:

- `op/fused_act.py`
- `tools/parse_face.py`
- `models/encoder4editing/models/stylegan2/op/fused_act.py`

Even where they were not the final blocking issue, they are part of the verified working state and should be preserved by using the known-good tags and commits listed above.

### Verified working environment

A new conda environment was created and verified:

- Environment name: `rephoto_cuda11`
- Python: `3.8`
- PyTorch: `2.4.1`
- CUDA runtime: `11.8`
- GPU verified: `NVIDIA GeForce RTX 3060 Laptop GPU`

Additional packages installed into this environment:

- `opencv`
- `tensorboard`
- `matplotlib`
- `tqdm`
- `scikit-image`

The exported environment file is included in the repo:

- `rephoto_cuda11_working.yml`

A patch snapshot was also generated during troubleshooting:

- `rephoto_cuda11_working.patch`

### Required local assets

These assets are required locally but may not exist in a fresh clone:

1. Test input image:
   - `dataset\Abraham Lincoln_01.png`

2. Main checkpoints:
   - `checkpoint\*`

3. Face parsing checkpoint:
   - `third_party\face_parsing\res\cp\79999_iter.pth`

### Bootstrap script for local assets

To simplify restore/setup, the repository includes:

- `bootstrap_local_assets.ps1`

Current usage:

    .\bootstrap_local_assets.ps1 -SourceRepo "C:\Users\james\Projects\Time-Travel-Rephotography.github.io"

This script copies:
- the verified test image into `.\dataset`
- the main checkpoint tree into `.\checkpoint`
- the face parsing checkpoint into `.\third_party\face_parsing\res\cp`

### Verified restore procedure

A clean restore test was successfully performed from a parent folder.

High-level process:

1. Clone the repository with submodules:
   - `git clone --recurse-submodules ...`

2. Check out the verified tag:
   - `rephoto_cuda11_working_2026-02-26_bootstrap_v2`

3. Confirm the submodule resolves to:
   - `9520744f95c9109c3cfcd5ca1f5f0dc6da70541f`

4. Run the bootstrap script with `-SourceRepo` pointing at a local working copy that already contains the needed assets.

5. Run projector successfully in the clean restore clone.

### Verified projector commands

Stage-1 run (32x32 only):

    python projector.py "dataset\Abraham Lincoln_01.png" `
      --encoder_ckpt "checkpoint\encoder\checkpoint_g.pt" `
      --color_transfer 0 `
      --eye 0 `
      --lr 0.001 `
      --noise_regularize 0 `
      --camera_lr 0 `
      --wplus_step 250 `
      --results_dir "results/projector_restore_test_v2"

Full run (32x32 + 64x64):

    python projector.py "dataset\Abraham Lincoln_01.png" `
      --encoder_ckpt "checkpoint\encoder\checkpoint_g.pt" `
      --color_transfer 0 `
      --eye 0 `
      --lr 0.001 `
      --noise_regularize 0 `
      --camera_lr 0 `
      --results_dir "results/projector_restore_test_full_v2"

### What a successful run now confirms

A successful run now confirms the full intended projector protocol is working:

1. Initialize latent code (encoder / e4e path)
2. Generate the initial sibling image
3. Run face parsing and skin-mask generation
4. Run histogram matching
5. Save the initial image and latent/noise state
6. Optimize through the configured W+ stages
7. Save the final image, final latent/noise `.pt`, and random-noise variant under the selected `results_dir`

### Practical recommendation

For Windows use, treat the following as the canonical reproducible state:

- main repo checked out at `rephoto_cuda11_working_2026-02-26_bootstrap_v2`
- `models/encoder4editing` submodule at `encoder4editing_cuda11_working_2026-02-25`
- conda environment restored from `rephoto_cuda11_working.yml`
- local assets restored with `bootstrap_local_assets.ps1`

This is the currently verified working baseline for the Windows CUDA 11.8 projector workflow.
