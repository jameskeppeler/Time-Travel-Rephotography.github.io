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

## This fork at a glance

This repository now contains **two related workflows**:

1. **Original upstream workflow** for the research projector.
2. **A Windows-focused, developer-friendly wrapper workflow** that adds:
   - automatic face cropping from raw images,
   - optional GFPGAN restoration,
   - conservative blend-back of restored faces,
   - a configurable projector runner for one or many cropped faces.

For the full Windows setup, asset bootstrapping steps, and wrapper workflow, see [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

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
