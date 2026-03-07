# RePhoto Parameter Guide

This document explains the main rephotography parameters used in the local Time-Travel-Rephotography workflow, how they affect image quality, and how the current Colab-matching setup differs from the earlier stripped-down local setup.

---

# Current Colab-Matching Parameter Set

```text
camera_lr: 0.01
ckpt: checkpoint/stylegan2-ffhq-config-f.pt
coarse_min: 32
color_transfer: 10000000000.0
contextual: 0.1
cx_layers: ['relu3_4', 'relu2_2', 'relu1_2']
e4e_ckpt: checkpoint/e4e_ffhq_encode.pt
e4e_size: 256
encoder_ckpt: checkpoint/encoder/checkpoint_b.pt
encoder_name: b
encoder_size: 256
eye: 0.1
gaussian: 0.75
generator_size: 1024
init_latent: None
input: dataset/Abraham Lincoln_01.png
log_dir: log/
log_freq: 10
log_visual_freq: 1000
lr: 0.1
mix_layer_range: [10, 18]
noise_ramp: 0.75
noise_regularize: 50000.0
noise_strength: 0.0
rand_seed: None
recon_size: 256
results_dir: results/
spectral_sensitivity: b
vgg: 1
vggface: 0.3
wplus_step: [250, 750]
```

---

# What Each Parameter Does

## Core model / checkpoints

### `ckpt`
Main StyleGAN2 generator checkpoint. This determines the face model the projector is optimizing into.

### `generator_size`
The native output size of the generator. In this project it is 1024.

### `e4e_ckpt`
Checkpoint for the e4e inversion encoder. This helps produce the starting latent code before optimization.

### `e4e_size`
Input resize used for e4e initialization.

### `encoder_ckpt`
Checkpoint for the auxiliary encoder used in initialization.

### `encoder_name`
Identifier for the auxiliary encoder family. The checkpoint path is the part that matters most in practice.

### `encoder_size`
Input resize used for the auxiliary encoder.

---

## Initialization / latent mixing

### `mix_layer_range`
Controls which layers of the e4e code are replaced by the auxiliary encoder code. This strongly affects the balance between identity structure and appearance information.

### `init_latent`
Optional manual latent starting point. `None` means normal encoder-based initialization is used.

### `rand_seed`
Optional random seed for reproducibility. `None` means runs are not forced to be exactly repeatable.

---

## Optimization schedule

### `wplus_step`
The optimization schedule. For example, `[250, 750]` means a two-stage run with 250 steps in the first stage and 750 in the second.

### `coarse_min`
Controls the starting coarse scale for staged optimization.

### `lr`
Learning rate for the main latent/noise optimization.

### `camera_lr`
Learning rate for the degradation/camera model parameters.

---

## Noise behavior

### `noise_strength`
Amount of latent noise injected during optimization. At `0.0`, it is effectively off.

### `noise_ramp`
How quickly the latent noise decays over the run. Since `noise_strength = 0.0`, this is mostly inactive in the current setup.

### `noise_regularize`
Penalty on pathological StyleGAN noise usage. Helps prevent the optimizer from hiding junk detail in noise maps.

---

## Loss terms

### `vgg`
General perceptual loss weight.

### `vggface`
Face-specific perceptual/identity preservation weight.

### `recon_size`
Resize used for perceptual reconstruction loss.

### `eye`
Extra loss on the eye region to help preserve eyes.

### `contextual`
Contextual loss weight. Helps match structure and appearance without forcing exact pixel identity.

### `cx_layers`
The VGG layers used by contextual loss.

### `color_transfer`
Strongly enforces tonal/rendering consistency with the target/sibling image.

---

## Historical-photo degradation model

### `spectral_sensitivity`
Controls how the system models historical photographic channel sensitivity. In this workflow, `b` is used for blue-sensitive behavior.

### `gaussian`
Blur radius in the degradation model. Helps the synthetic output match the softness of historical source material.

---

## Input / output / logging

### `input`
Input image path.

### `results_dir`
Directory where outputs are written.

### `log_dir`
Directory where log data is written.

### `log_freq`
How often scalar values are logged.

### `log_visual_freq`
How often image snapshots are logged.

---

# How This Differs From the Original Stripped-Down Local Setup

The earlier stripped-down local wrapper was not a completely different system, but it explicitly disabled or weakened several important terms.

## Important differences

### `lr`
- Current Colab-style: `0.1`
- Earlier stripped-down: `0.001`

The earlier setup optimized much more cautiously.

### `camera_lr`
- Current: `0.01`
- Earlier: `0`

Earlier, the degradation model was effectively frozen.

### `color_transfer`
- Current: `10000000000.0`
- Earlier: `0`

Earlier, tonal/rendering transfer was disabled.

### `eye`
- Current: `0.1`
- Earlier: `0`

Earlier, there was no explicit eye preservation term.

### `noise_regularize`
- Current: `50000`
- Earlier: `0`

Earlier, there was no strong penalty against pathological noise usage.

### `gaussian`
- Current: `0.75`
- Earlier: effectively `0`

Earlier, historical blur was not modeled explicitly.

### `spectral_sensitivity`
- Current: `b`
- Earlier local default: `g`

Earlier, the degradation model was more generic.

### `log_visual_freq`
- Current: `1000`
- Earlier local default: `50`

This mainly affects logging overhead, not final image quality directly.

### `encoder_ckpt`
- Current target parity: `checkpoint_b.pt`
- Earlier local runs often used: `checkpoint_g.pt`

This changes the initialization behavior.

---

# Which Parameters Matter Most for Image Quality

## Overall ranking

1. `wplus_step`
2. `ckpt`
3. `encoder_ckpt`
4. `e4e_ckpt`
5. `mix_layer_range`
6. `color_transfer`
7. `vggface`
8. `vgg`
9. `contextual`
10. `cx_layers`
11. `spectral_sensitivity`
12. `gaussian`
13. `eye`
14. `lr`
15. `camera_lr`
16. `noise_regularize`

Everything below that is secondary, inactive in the current setup, or mostly administrative.

---

# Which Parameters Matter Most for Identity Preservation

## Highest impact
- `vggface`
- `e4e_ckpt`
- `encoder_ckpt`
- `mix_layer_range`
- `wplus_step`

## Next tier
- `vgg`
- `eye`
- `lr`

## Identity summary
If the problem is "this does not look like the same person," focus first on:
1. `vggface`
2. `e4e_ckpt`
3. `encoder_ckpt`
4. `mix_layer_range`
5. `wplus_step`

---

# Which Parameters Matter Most for Historical Look

## Highest impact
- `spectral_sensitivity`
- `gaussian`
- `color_transfer`
- `contextual`
- `cx_layers`

## Next tier
- `camera_lr`
- `mix_layer_range`
- `coarse_min`

## Historical-look summary
If the problem is "this looks too modern or too clean," focus first on:
1. `spectral_sensitivity`
2. `gaussian`
3. `color_transfer`
4. `contextual`
5. `cx_layers`

---

# Which Parameters Matter Most for Artifact Reduction / Stability

## Highest impact
- `noise_regularize`
- `lr`
- `wplus_step`
- `eye`

## Next tier
- `vgg`
- `vggface`
- `camera_lr`
- `gaussian`

## Artifact summary
If the problem is "this has weird textures, broken eyes, or unstable detail," focus first on:
1. `noise_regularize`
2. `lr`
3. `wplus_step`
4. `eye`

---

# Practical Tuning Cheat Sheet

## If identity is weak
Try adjusting:
- `wplus_step`
- `vggface`
- `mix_layer_range`

## If the result looks too modern
Try adjusting:
- `spectral_sensitivity`
- `gaussian`
- `color_transfer`
- `contextual`

## If the result has artifacts
Try adjusting:
- `noise_regularize`
- `lr`
- `wplus_step`
- `eye`

---

# Short Version

## Most important user-facing quality knob
- `wplus_step`

## Most important initialization knobs
- `encoder_ckpt`
- `e4e_ckpt`
- `mix_layer_range`

## Most important loss/appearance knobs
- `vggface`
- `vgg`
- `contextual`
- `color_transfer`
- `eye`

## Most important historical rendering knobs
- `spectral_sensitivity`
- `gaussian`
- `camera_lr`

## Most important stability knobs
- `noise_regularize`
- `lr`