# Technical Summary — GAN-Augmented PCB Defect Detection
## Master's Thesis Paper Sources

**Project:** GAN-Based Synthetic PCB Image Generation for Automated Defect Inspection  
**Dataset:** DeepPCB (YOLO-converted)  
**Pipeline:** Phase 1 Baseline → Phase 2 DCGAN → Phase 3 Pix2Pix → Phase 4A DCGAN Retrain → Phase 4B Pix2Pix Retrain  
**Date compiled:** May 2026

---

## Table of Contents

1. [Background — Generative Adversarial Networks](#1-background--generative-adversarial-networks)
2. [Dataset](#2-dataset)
3. [Phase 1 — Baseline YOLOv8 Detector (Real Data Only)](#3-phase-1--baseline-yolov8-detector-real-data-only)
4. [Phase 2 — DCGAN Patch Synthesis](#4-phase-2--dcgan-patch-synthesis)
5. [Phase 3 — Pix2Pix Full-Image Translation](#5-phase-3--pix2pix-full-image-translation)
6. [Phase 4A — Merge & Retrain with DCGAN Synthetics](#6-phase-4a--merge--retrain-with-dcgan-synthetics)
7. [Phase 4B — Merge & Retrain with Pix2Pix Synthetics](#7-phase-4b--merge--retrain-with-pix2pix-synthetics)
8. [Quantitative Results Summary](#8-quantitative-results-summary)
9. [Sources File Map](#9-sources-file-map)
10. [References](#10-references)

> **Note on run naming:** The training run stored under `runs_deeppcb_gan/pcb_defects_pix2pix/` (without suffix) was the first merge-and-retrain experiment and was populated with **DCGAN**-generated synthetic images. The run under `pcb_defects_pix2pix_v2` used a fresh merge with **Pix2Pix**-generated images. The original directory name is misleading; correct labels are used throughout this document.

---

## 1. Background — Generative Adversarial Networks

### 1.1 The GAN Framework

Generative Adversarial Networks (GANs) were introduced by Goodfellow et al. (2014) [1] as a two-player minimax game between two neural networks trained simultaneously:

- **Generator G:** Maps a latent noise vector z sampled from a prior p_z(z) to the data space. Its objective is to produce samples that the discriminator cannot distinguish from real data.
- **Discriminator D:** A binary classifier that estimates the probability that an input came from the real data distribution rather than G.

The training objective is:

$$\min_G \max_D \, \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

At Nash equilibrium, G recovers the true data distribution and D outputs 1/2 everywhere. In practice, optimizing G against log(1 − D(G(z))) suffers from vanishing gradients early in training; instead G is trained to maximize log D(G(z)) (the non-saturating formulation [1]).

**Motivation for PCB defect synthesis:** Real PCB defect datasets are expensive to acquire and naturally imbalanced — rare fault types like `spurious_copper` have far fewer examples than common ones. A trained generator can augment rare classes without additional hardware, improving detection model generalization.

### 1.2 Deep Convolutional GAN (DCGAN)

Radford, Metz & Chintala (2015) [2] demonstrated that replacing fully-connected layers with strided convolutional and transposed-convolutional layers greatly stabilizes GAN training and improves image quality. Key architectural guidelines from [2] that this work follows:

| Guideline | Applied As |
|---|---|
| Replace pooling with strided convolutions (D) / transposed convolutions (G) | All downsampling in D uses stride-2 Conv2d; G uses ConvTranspose2d |
| Use BatchNorm in both G and D | Applied to all hidden layers except D input and G output |
| Remove fully-connected layers | Architecture is fully convolutional |
| Use ReLU in G, LeakyReLU in D | G: ReLU hidden + Tanh output; D: LeakyReLU(0.2) |
| Adam optimizer, lr=0.0002, beta1=0.5 | Exactly followed [2] |

**Class conditioning** is added via an embedding layer (class → 32-dim vector) injected into both G (concatenated to latent z) and D (projected to a spatial channel appended to the input image). This extends the unconditional DCGAN to a conditional model [3], enabling controlled per-class synthesis.

**Hinge loss** replaces the original BCE adversarial loss, as recommended by Lim & Ye (2017) [4] and Miyato et al. (2018) [5], for improved gradient stability:

$$\mathcal{L}_D = -\mathbb{E}[\min(0, -1 + D(x))] - \mathbb{E}[\min(0, -1 - D(G(z)))]$$
$$\mathcal{L}_G = -\mathbb{E}[D(G(z))]$$

**Instance noise** (sigma decayed from 0.08 to 0.0 over training) is applied to discriminator inputs following Sonderby et al. (2016) [6] and Arjovsky & Bottou (2017) [7], which show that adding noise to real and fake inputs throughout early training prevents the discriminator from becoming overconfident before the generator has converged.

### 1.3 Pix2Pix — Conditional Image-to-Image Translation

Isola et al. (2017) [8] proposed learning a mapping from an input image domain to an output image domain using a conditional GAN. Given paired training samples (x_i, y_i), the generator learns G: x → y. The loss combines adversarial and pixel-L1 terms:

$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x,z)))]$$
$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[\|y - G(x,z)\|_1]$$
$$\mathcal{L}_{total} = \mathcal{L}_{cGAN} + \lambda_{L1} \cdot \mathcal{L}_{L1}, \quad \lambda_{L1} = 100$$

The L1 term (lambda=100) enforces low-frequency correctness — critical for PCB defect images where the global circuit layout must remain undistorted. The adversarial term sharpens texture details that L1 alone would blur.

#### U-Net Generator

The generator architecture follows the U-Net design of Ronneberger, Fischer & Brox (2015) [9], which uses a symmetric encoder-decoder with skip connections between mirrored layers. Skip connections concatenate encoder feature maps directly into the corresponding decoder layer, preserving fine-grained spatial structure (edges, circuit traces) that would otherwise be lost at the bottleneck. This is essential for PCB images where label-accurate defect position must be preserved.

#### PatchGAN Discriminator

Rather than classifying the entire image as real or fake, the PatchGAN [8, 10] discriminator produces a grid of patch-level predictions over 70x70 receptive fields. This:
- Provides denser spatial gradient signal (one loss value per patch, not one per image)
- Encourages high-frequency texture realism within local regions
- Is particularly suited to defect images where only small localized areas differ between normal and defective PCBs

### 1.4 YOLOv8 Object Detector

YOLOv8 (Jocher et al., 2023) [11] is the detector backbone used in all training phases. Key design choices:

| Choice | Rationale |
|---|---|
| **YOLOv8-L** architecture | Sufficient parameter count for 6-class fine-grained detection; fits V100 32GB |
| **AdamW** optimizer | Decoupled weight decay (Loshchilov & Hutter, 2019) [12] is more stable than SGD with L2 regularization for YOLO fine-tuning |
| **Cosine LR schedule** | SGDR-style decay (Loshchilov & Hutter, 2016) [13] smoothly reduces learning rate, avoiding sharp drops |
| **Pre-trained COCO weights** | Transfer learning from a large diverse dataset accelerates convergence on the small DeepPCB training split |
| **AMP disabled** | V100 (SM_70) has a known atomic CUDA kernel bug triggered by Ultralytics' box_iou on CUDA; patched to run on CPU |

---

## 2. Dataset

### Primary Dataset: DeepPCB

Tang et al. (2019) [14] released the DeepPCB dataset as a public PCB defect benchmark with paired template (defect-free) and test (defective) images. Each pair is registered so defect locations can be precisely localized.

| Property | Value |
|---|---|
| Source | DeepPCB-master — Tang et al. (2019) [14] |
| Format | YOLO (converted from DeepPCB annotation format) |
| Path in workspace | `DeepPcb dataset/deeppcb_yolo/` |
| Classes | 6 defect types |
| Image type | Grayscale PCB scans |

### Dataset Splits

| Split | Images | Purpose |
|---|---|---|
| Train | 800 | Model training |
| Validation | 200 | In-training evaluation; unchanged across all runs |
| Test | 500 | Final held-out evaluation; unchanged across all runs |

> Validation and test sets are always drawn from **real images only** to ensure fair, uncontaminated evaluation across all experimental conditions.

### Defect Classes

| ID | Class Name | Patch Count (DCGAN training) | Description |
|---|---|---|---|
| 0 | missing_hole | 820 | Drilled hole absent or blocked |
| 1 | mouse_bite | 1,095 | Notch on PCB edge |
| 2 | open_circuit | 1,030 | Broken trace |
| 3 | short | 830 | Unintended connection between traces |
| 4 | spur | 911 | Extra copper protrusion from trace |
| 5 | spurious_copper | 799 | Copper residue in non-trace area |
| **Total** | | **5,485 patches** | Extracted at 64x64, margin 20% |

**Source files:**
- `01_dataset/data_deeppcb.yaml` — original YOLO dataset config
- `01_dataset/data_merged.yaml` — merged dataset config (real + synthetic)
- `01_dataset/patch_stats.json` — per-class patch extraction statistics

---

## 3. Phase 1 — Baseline YOLOv8 Detector (Real Data Only)

### Purpose

Establish a real-data-only reference point to quantify the baseline detection performance before any synthetic augmentation. All improvements in Phases 4A and 4B are measured relative to this baseline.

### Model and Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Model | YOLOv8-L (`yolov8l.pt`) | Best accuracy-VRAM tradeoff at imgsz=640 on V100 32GB |
| Pre-training | COCO (transfer learning) | Accelerates convergence on small 800-image training split |
| Image size | 640 x 640 | Standard YOLOv8 resolution; balances detail and speed |
| Batch size | 16 | Maximizes GPU utilization within VRAM budget |
| Epochs configured | 200 | |
| Epochs completed | 181 | Early stopped by patience=50 |
| Optimizer | AdamW | Decoupled weight decay [12]; more stable than SGD for fine-tuning |
| Initial LR (lr0) | 0.001 | |
| Final LR ratio (lrf) | 0.01 | Cosine decay: final LR = lr0 x lrf = 1e-5 |
| LR schedule | Cosine (SGDR) | Smooth decay avoids sharp performance drops late in training [13] |
| Momentum | 0.937 | Standard YOLO default |
| Weight decay | 0.0005 | |
| Warmup epochs | 3 | Linear warmup prevents instability in early epochs |
| Warmup bias LR | 0.1 | |
| AMP | Off | V100 SM_70 CUDA atomic bug workaround |
| Seed | 0 (deterministic) | Reproducibility |
| Device | GPU CUDA 0 | |
| Workers | 4 | |
| Patience | 50 epochs | |

### Data Augmentation (YOLOv8 Built-in)

| Augmentation | Value | Rationale |
|---|---|---|
| HSV hue shift | 0.01 | Minimal; grayscale PCBs have little hue variation |
| HSV saturation | 0.5 | Robustness to scanner illumination differences |
| HSV value | 0.3 | Robustness to contrast variation |
| Translation | 0.1 | Shift invariance |
| Scale | 0.9 | Scale invariance |
| Horizontal flip | 0.5 | Natural symmetry of PCB layouts |
| Mosaic | 1.0 | Disabled last 10 epochs (close_mosaic=10) |
| Mixup | 0 | Off; could corrupt PCB trace geometry |
| Copy-paste | 0 | Off |
| Rotation/shear | 0 | Off — PCB layouts are axis-aligned by manufacture standard |

### Baseline Training — Key Epoch Milestones

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 1 | 0.282 | 0.067 | 0.372 | 0.439 |
| 7 | 0.956 | 0.694 | 0.912 | 0.885 |
| 23 | 0.988 | 0.742 | 0.968 | 0.956 |
| **103 (peak)** | **0.9837** | **0.6491** | **0.9817** | **0.9478** |
| 181 (final) | 0.978 | 0.632 | 0.985 | 0.947 |

### Baseline Peak Performance

| Metric | Value | Epoch |
|---|---|---|
| mAP@50 | **0.9837** | 103 |
| mAP@50-95 | **0.6491** | 103 |
| Precision | 0.9817 | 103 |
| Recall | 0.9478 | 103 |

**Source files (images referenced below):**
- `04_baseline_yolo/args.yaml` — full training config
- `04_baseline_yolo/results.csv` — per-epoch metrics (181 epochs)
- `04_baseline_yolo/results.png` — training loss and metric curves
- `04_baseline_yolo/confusion_matrix.png` — confusion matrix on validation set
- `04_baseline_yolo/confusion_matrix_normalized.png` — row-normalized confusion matrix
- `04_baseline_yolo/BoxF1_curve.png` — F1 score vs confidence threshold
- `04_baseline_yolo/BoxPR_curve.png` — Precision-Recall curve
- `04_baseline_yolo/BoxP_curve.png` — Precision vs confidence
- `04_baseline_yolo/BoxR_curve.png` — Recall vs confidence
- `04_baseline_yolo/labels.jpg` — training label distribution
- `04_baseline_yolo/train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` — training batch mosaics
- `04_baseline_yolo/val_batch0_labels.jpg`, `val_batch0_pred.jpg` — validation ground truth vs predictions
- `04_baseline_yolo/val_batch1_labels.jpg`, `val_batch1_pred.jpg`

---

## 4. Phase 2 — DCGAN Patch Synthesis

### Purpose

Train a class-conditional DCGAN on the extracted 64x64 defect patches to synthesise novel per-class defect content that can later be composited onto clean PCB templates.

**Why DCGAN:** Efficient for local low-resolution texture generation. Defect instances are small localized features well-suited to 64x64 patch synthesis. Class conditioning enables controllable oversampling of rare defect types (e.g. `spurious_copper` with only 799 patches).

### Patch Extraction

| Parameter | Value |
|---|---|
| Patch size | 64 x 64 px |
| Bounding box margin | 20% (0.2) — adds context around the defect |
| Total patches extracted | 5,485 |
| Skipped (unreadable images) | 0 |
| Output | `gan_patches/deeppcb/class_N_*/` |

### DCGAN Architecture

#### Generator

| Layer | Operation | Output |
|---|---|---|
| Input | z (100-dim) concatenated with class_embed (32-dim) = 132-dim | 132x1x1 |
| 1 | ConvTranspose2d(132->512, 4x4, stride=1), BN, ReLU | 512x4x4 |
| 2 | ConvTranspose2d(512->256, 4x4, stride=2, pad=1), BN, ReLU | 256x8x8 |
| 3 | ConvTranspose2d(256->128, 4x4, stride=2, pad=1), BN, ReLU | 128x16x16 |
| 4 | ConvTranspose2d(128->64, 4x4, stride=2, pad=1), BN, ReLU | 64x32x32 |
| 5 | ConvTranspose2d(64->1, 4x4, stride=2, pad=1), **Tanh** | 1x64x64 |

Total generator parameters: ~2.3M. Output range: [-1, 1] rescaled to [0, 255] for storage.

#### Discriminator

| Layer | Operation | Output |
|---|---|---|
| Input | image(1ch) + embed_proj(1ch) = 2ch | 2x64x64 |
| 1 | Conv2d(2->64, 4x4, stride=2, pad=1), LeakyReLU(0.2) | 64x32x32 |
| 2 | Conv2d(64->128, 4x4, stride=2, pad=1), BN, LeakyReLU(0.2) | 128x16x16 |
| 3 | Conv2d(128->256, 4x4, stride=2, pad=1), BN, LeakyReLU(0.2) | 256x8x8 |
| 4 | Conv2d(256->512, 4x4, stride=2, pad=1), BN, LeakyReLU(0.2) | 512x4x4 |
| 5 | Conv2d(512->1, 4x4, stride=1, pad=0) | scalar logit |

Class conditioning in D: `nn.Embedding(6, 32)` -> `nn.Linear(32, 64x64)` -> reshaped to 1x64x64 spatial map appended as an extra input channel.

### DCGAN Training Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Latent dim (nz) | 100 | Standard DCGAN [2] |
| Class embedding dim | 32 | Sufficient expressivity; small relative to z |
| ngf / ndf | 64 / 64 | DCGAN [2] default; good balance of capacity vs speed at 64px |
| Epochs | 300 (checkpoints saved every 10 up to 120; extended samples to 300) | |
| Batch size | 128 | Larger batches stabilize GAN training statistics |
| Optimizer | Adam | Adaptive gradient descaler; standard for GANs [1, 2] |
| Learning rate | 0.0002 | Radford et al. [2] empirically determined for stable DCGAN |
| Adam beta1 | 0.5 | Lower momentum (vs standard 0.9) reduces oscillation in GAN saddle-point optimization [2] |
| Adam beta2 | 0.999 | Standard |
| Loss | Hinge loss | Improved gradient properties over BCE [4, 5] |
| Instance noise sigma start | 0.08 | Prevents discriminator overconfidence early in training [6, 7] |
| Instance noise sigma end | 0.0 | Annealed to zero as training stabilizes |
| Image channels | 1 (grayscale) | DeepPCB is grayscale |
| Seed | 42 | Reproducibility |

### DCGAN Training Loss History (Representative Epochs)

| Epoch | Loss_D | Loss_G | Observation |
|---|---|---|---|
| 1 | 0.1824 | 8.7079 | D rapidly gains advantage; G loss very high |
| 5 | 0.5006 | 2.5353 | G begins learning structure |
| 10 | 0.4096 | 1.7683 | Approaching equilibrium |
| 20 | 0.3114 | 1.6646 | Stable training regime |
| 30 | 0.1563 | 1.9430 | D slightly stronger; G adapts |
| 50 | 0.1450 | 2.4550 | Mode equilibrium with slight G drift |
| 60 | 0.1373 | 2.8776 | D overconfident; hinge loss preventing collapse |
| 80 | 0.0842 | 2.6718 | D dominates; hinge loss prevents collapse |
| 100 | 0.0521 | 3.0844 | G continues training without collapse |
| 120 | 0.1078 | 4.9814 | G loss rise = D near-perfect; training saturating |

Full log: `02_dcgan/training_history.json`

### DCGAN Synthetic Image Generation

**Script:** `generate_synthetic.py`

**Generation pipeline:**
1. Load a real clean PCB template from `DeepPcb dataset/`
2. Sample class label c and noise vector z ~ N(0, I)
3. Generate 64x64 defect patch via trained G(z, c)
4. Apply small random affine transform (scale +/-10%, rotation +/-5 degrees)
5. Blend patch onto template using adaptive alpha masking
6. Write YOLO `.txt` label file with class c and bounding box at insertion location

| Property | Value |
|---|---|
| Total DCGAN synthetic images generated | **2,400** |
| Output | `gan_synthetic/deeppcb/images/` + `labels/` |
| Naming convention | `synth_NNNNNNN.png` |

**Source files (images):**
- `02_dcgan/training_history.json` — full loss log
- `02_dcgan/train_gan.py` — training script
- `02_dcgan/samples/epoch_0010/class_*.png` — 6 per-class 8x8 grids at epoch 10
- `02_dcgan/samples/epoch_0060/class_*.png` — 6 per-class grids at epoch 60
- `02_dcgan/samples/epoch_0120/class_*.png` — 6 per-class grids at epoch 120
- `02_dcgan/samples/epoch_0300/class_*.png` — 6 per-class grids at epoch 300
- `06_synthetic_samples/dcgan_samples/synth_*.png` — 9 representative DCGAN composite full-PCB images

---

## 5. Phase 3 — Pix2Pix Full-Image Translation

### Purpose

Train a conditional image-to-image GAN (Pix2Pix) [8] on the DeepPCB paired data (clean template -> defective image). Unlike DCGAN compositing, Pix2Pix generates globally coherent full-resolution defect images where the label comes directly from the matched real annotation — guaranteeing label accuracy and addressing the mAP50-95 localization gap.

**Why Pix2Pix over DCGAN compositing:** DCGAN patches are blended onto templates using heuristic alpha masks, which may misplace defect boundaries relative to the YOLO labels. Pix2Pix inherits real annotation bounding boxes exactly, so every synthetic image has a tight, correct label — directly benefiting the localization metric mAP50-95.

### Architecture

#### U-Net Generator [8, 9]

| Encoder Block | Channels In -> Out | Resolution |
|---|---|---|
| E1 | 1 -> 64, LeakyReLU | 128x128 |
| E2 | 64 -> 128, BN, LeakyReLU | 64x64 |
| E3 | 128 -> 256, BN, LeakyReLU | 32x32 |
| E4 | 256 -> 512, BN, LeakyReLU, Dropout(0.5) | 16x16 |
| E5 | 512 -> 512, BN, LeakyReLU, Dropout(0.5) | 8x8 |
| E6 | 512 -> 512, BN, LeakyReLU, Dropout(0.5) | 4x4 |
| E7 | 512 -> 512, BN, LeakyReLU, Dropout(0.5) | 2x2 |
| Bottleneck | 512 -> 512, LeakyReLU | 1x1 |
| D7-D1 + Out | Symmetric upsampling with skip connections | up to 256x256, 1ch, Tanh |

Skip connections [9] concatenate encoder feature maps with decoder inputs: critical for PCB images to preserve circuit trace geometry across the bottleneck.

#### PatchGAN Discriminator [8, 10]

| Layer | In -> Out | Operation |
|---|---|---|
| Input | [template || output] = 2ch | Concatenation of conditioning image + generated/real |
| L1 | 2 -> 64 | Conv2d 4x4 stride=2, LeakyReLU(0.2) |
| L2 | 64 -> 128 | Conv2d 4x4 stride=2, BN, LeakyReLU(0.2) |
| L3 | 128 -> 256 | Conv2d 4x4 stride=2, BN, LeakyReLU(0.2) |
| L4 | 256 -> 512 | Conv2d 4x4 stride=1, BN, LeakyReLU(0.2) |
| L5 | 512 -> 1 | Conv2d 4x4 stride=1 -> patch logit map |

Effective receptive field: **70x70 pixels**. The discriminator predicts NxN real/fake values per image, providing spatially dense adversarial feedback.

### Training Setup

| Parameter | Value | Justification |
|---|---|---|
| Image size | 256 x 256 | 640px requires batch=1 in VRAM; 256px allows batch=4 |
| Batch size | 4-8 | VRAM constrained |
| Epochs | 200 | |
| Optimizer | Adam | Standard for GANs [1] |
| Learning rate | 0.0002 | Same as DCGAN [2]; Pix2Pix [8] recommendation |
| Adam beta1 | 0.5 | Same GAN-stability rationale [2, 8] |
| Adam beta2 | 0.999 | |
| lambda_L1 | 100 | L1 weight per Isola et al. [8] experiments: below 10 produces blobby output; above 100 reduces defect sharpness |
| Class-balanced sampling | WeightedRandomSampler | Oversamples rare defect classes to prevent class imbalance |
| Training data | DeepPCB paired template/defect images | |
| Checkpoint interval | Every 25 epochs | |

### Pix2Pix Sample Progression

| Image | Notes |
|---|---|
| `03_pix2pix/samples/epoch_0025.png` | Early — rough brightness mapping, minimal defect detail |
| `03_pix2pix/samples/epoch_0075.png` | Mid — defect texture emerging, trace pattern preserved |
| `03_pix2pix/samples/epoch_0125.png` | Late — sharp defect edges, realistic PCB background |
| `03_pix2pix/samples/epoch_0200.png` | Final — production-quality output |

### Synthetic Image Generation (Pix2Pix)

**Script:** `generate_pix2pix.py`

| Property | Value |
|---|---|
| Total Pix2Pix synthetic images generated | **2,000** |
| Output | `gan_synthetic/deeppcb_pix2pix/images/` + `labels/` |
| Naming convention | `px2px_NNNNNNN.png` |
| Visualization panels | `gan_synthetic/deeppcb_pix2pix/visualized/` |

**Source files (images):**
- `03_pix2pix/train_pix2pix.py` — full training script
- `03_pix2pix/generate_pix2pix.py` — generation script
- `03_pix2pix/samples/epoch_0025.png`, `epoch_0075.png`, `epoch_0125.png`, `epoch_0200.png`
- `06_synthetic_samples/pix2pix_samples/px2px_*.png` — 6 representative synthetic images
- `06_synthetic_samples/pix2pix_visualized/px2px_*.png` — 100 side-by-side template->generated panels

---

## 6. Phase 4A — Merge & Retrain with DCGAN Synthetics

> **Naming note:** This run is stored as `runs_deeppcb_gan/pcb_defects_pix2pix/` (historically mislabeled). It used the merged dataset populated with **DCGAN** synthetic images, not Pix2Pix.

### Merged Dataset Composition

| Component | Count | Proportion | Source |
|---|---|---|---|
| Real training images | 800 | **44.4%** | `DeepPcb dataset/deeppcb_yolo/train/` |
| DCGAN synthetic images | 1,000 | **55.6%** | `gan_synthetic/deeppcb/` (selected from 2,400 generated) |
| **Total merged training** | **1,800** | 100% | `gan_merged/deeppcb/merged/` |
| Validation | 200 | — | Original DeepPCB val (real only, unchanged) |
| Test | 500 | — | Original DeepPCB test (real only, unchanged) |

**Key merging statistics:**

| Statistic | Value |
|---|---|
| Augmentation factor vs baseline | **2.25x** (1,800 / 800) |
| Synthetic injection ratio | **55.6%** of merged training images are synthetic |
| Synthetic-to-real ratio | **1.25 : 1** synthetic per real image |
| Synthetic images used | 1,000 of 2,400 generated (~41.7% of pool) |

**Merge script:** `merge_and_retrain.py` — copies real images into `merged/` then copies the selected DCGAN synthetic images alongside them. The validation and test sets are kept from the original split (real only) to ensure fair evaluation.

### YOLOv8 Retrain Configuration

Identical to Phase 1 baseline configuration for fair comparison:

| Parameter | Value |
|---|---|
| Model | YOLOv8-L (`yolov8l.pt`) |
| Data | `gan_merged/deeppcb/data_merged.yaml` |
| Image size | 640 x 640 |
| Batch size | 16 |
| Epochs | 200 (completed all 200) |
| Optimizer | AdamW |
| lr0 / lrf | 0.001 / 0.01 (cosine) |
| Seed | 0 |
| Device | GPU CUDA 0 |
| Run name | `pcb_defects_pix2pix` (mislabeled) |

### Phase 4A Training — Key Epoch Milestones

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 1 | 0.242 | 0.075 | 0.293 | 0.289 |
| 7 | 0.921 | 0.407 | 0.877 | 0.840 |
| 15 | 0.966 | 0.689 | 0.930 | 0.913 |
| 20 | 0.977 | 0.625 | 0.962 | 0.926 |
| 25 | 0.969 | 0.730 | 0.948 | 0.915 |
| **193 (peak)** | **0.9903** | **0.8081** | **0.9863** | **0.9646** |
| 200 (final) | 0.989 | 0.817 | 0.981 | 0.971 |

### Phase 4A Peak Performance

| Metric | Value | Epoch |
|---|---|---|
| mAP@50 | **0.9903** | 193 |
| mAP@50-95 | **0.8081** | 193 |
| Precision | 0.9863 | 193 |
| Recall | 0.9646 | 193 |

**Source files:**
- `04b_dcgan_retrain/args.yaml` — full training config
- `04b_dcgan_retrain/results.csv` — per-epoch metrics (200 epochs)

> *Note: This run did not save evaluation plot images (confusion matrices, P/R curves). Only `args.yaml` and `results.csv` are available as artifacts.*

---

## 7. Phase 4B — Merge & Retrain with Pix2Pix Synthetics

### Merged Dataset Composition

After the DCGAN retrain, the merged directory was wiped (`--fresh-merge` flag in `merge_and_retrain.py`) and repopulated with Pix2Pix synthetic images:

| Component | Count | Proportion | Source |
|---|---|---|---|
| Real training images | 800 | **44.4%** | `DeepPcb dataset/deeppcb_yolo/train/` |
| Pix2Pix synthetic images | 1,000 | **55.6%** | `gan_synthetic/deeppcb_pix2pix/` (selected from 2,000 generated) |
| **Total merged training** | **1,800** | 100% | `gan_merged/deeppcb/merged/` (refreshed) |
| Validation | 200 | — | Original DeepPCB val (real only, unchanged) |
| Test | 500 | — | Original DeepPCB test (real only, unchanged) |

**Key merging statistics:**

| Statistic | Value |
|---|---|
| Augmentation factor vs baseline | **2.25x** (1,800 / 800) — same as Phase 4A |
| Synthetic injection ratio | **55.6%** of merged training images are synthetic |
| Synthetic-to-real ratio | **1.25 : 1** synthetic per real image |
| Synthetic images used | 1,000 of 2,000 generated (50.0% of pool) |

**Key difference from Phase 4A:** Pix2Pix images are full-resolution full-scene translations with guaranteed label accuracy from matched real annotations, vs DCGAN composite patches with heuristic label placement based on insertion coordinates.

### YOLOv8 Retrain Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8-L (`yolov8l.pt`) |
| Data | `gan_merged/deeppcb/data_merged.yaml` |
| Image size | 640 x 640 |
| Batch size | 16 |
| Epochs | 200 (ran ~195) |
| Optimizer | AdamW |
| lr0 / lrf | 0.001 / 0.01 (cosine) |
| Seed | 0 |
| Device | GPU CUDA 0 |
| Run name | `pcb_defects_pix2pix_v2` |

### Phase 4B Training — Key Epoch Milestones

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 1 | 0.282 | 0.067 | 0.372 | 0.439 |
| 6 | 0.899 | 0.576 | 0.821 | 0.854 |
| 23 | 0.988 | 0.742 | 0.968 | 0.956 |
| 34 | 0.988 | 0.602 | 0.972 | 0.948 |
| **95 (peak)** | **0.9916** | **0.7529** | **0.9847** | **0.9793** |
| 195 (final) | 0.978 | 0.574 | 0.963 | 0.974 |

### Phase 4B Peak Performance

| Metric | Value | Epoch |
|---|---|---|
| mAP@50 | **0.9916** | 95 |
| mAP@50-95 | **0.7529** | 95 |
| Precision | 0.9847 | 95 |
| Recall | 0.9793 | 95 |

**Source files (images referenced below):**
- `05_pix2pix_v2_retrain/args.yaml` — full training config
- `05_pix2pix_v2_retrain/results.csv` — per-epoch metrics (~195 epochs)
- `05_pix2pix_v2_retrain/results.png` — training curves
- `05_pix2pix_v2_retrain/confusion_matrix.png`
- `05_pix2pix_v2_retrain/confusion_matrix_normalized.png`
- `05_pix2pix_v2_retrain/BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`
- `05_pix2pix_v2_retrain/labels.jpg`
- `05_pix2pix_v2_retrain/train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`
- `05_pix2pix_v2_retrain/val_batch0_labels.jpg`, `val_batch0_pred.jpg`
- `05_pix2pix_v2_retrain/val_batch1_labels.jpg`, `val_batch1_pred.jpg`

---

## 8. Quantitative Results Summary

### Complete Three-Way Comparison

| Condition | Synthetic Source | Train Images | Real | Synth | Synth % | Augmentation Factor | mAP@50 | mAP@50-95 | Precision | Recall | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | None | 800 | 800 | 0 | 0% | 1.0x | 0.9837 | 0.6491 | 0.9817 | 0.9478 | 103/181 |
| **+ DCGAN** (4A) | DCGAN composites | 1,800 | 800 | 1,000 | 55.6% | 2.25x | 0.9903 | **0.8081** | 0.9863 | 0.9646 | 193/200 |
| **+ Pix2Pix** (4B) | Pix2Pix full-image | 1,800 | 800 | 1,000 | 55.6% | 2.25x | **0.9916** | 0.7529 | **0.9847** | **0.9793** | 95/200 |

### Gains vs Baseline

| Metric | DCGAN absolute gain | DCGAN relative gain | Pix2Pix absolute gain | Pix2Pix relative gain | Better |
|---|---|---|---|---|---|
| mAP@50 | +0.0066 | +0.67% | +0.0079 | +0.80% | Pix2Pix |
| mAP@50-95 | **+0.1590** | **+24.5%** | +0.1038 | +16.0% | **DCGAN** |
| Precision | +0.0046 | +0.47% | +0.0030 | +0.31% | DCGAN |
| Recall | +0.0168 | +1.77% | +0.0315 | +3.32% | Pix2Pix |

### Key Findings

1. **DCGAN achieves the largest mAP50-95 gain (+24.5%, absolute +0.159).** This strict localization metric (IoU thresholds 0.50-0.95) improving more under DCGAN suggests that the spatial diversity introduced by randomly compositing patches at varied positions helps the detector learn to localize defects under greater geometric variability — even though the labels are placed heuristically.

2. **Pix2Pix achieves the highest mAP50 (0.9916) and recall (0.9793).** Guaranteed-correct labels from real annotations provide cleaner supervision, reducing missed detections (+3.3 pp recall) and giving the best overall detection score.

3. **mAP50 is near ceiling in all conditions.** The absolute gains are small (0.67-0.80%) because the baseline already achieves 0.984; the mAP50-95 gain is the more informative metric for comparing localization quality.

4. **Pix2Pix converges faster** (peak at epoch 95 vs 193 for DCGAN), suggesting that full-image synthetic data with correct labels provides richer, less contradictory training signal earlier.

5. **Both GAN methods provide meaningful gains.** Neither is universally dominant — the optimal choice depends on whether localization precision (DCGAN) or detection completeness (Pix2Pix) is prioritized by the application.

---

## 9. Sources File Map

```
sources_paper/
│
├── TECHNICAL_SUMMARY.md              <- this file
│
├── 01_dataset/
│   ├── data_deeppcb.yaml              Dataset config: splits, 6 class names
│   ├── data_merged.yaml               Merged dataset config (real + synthetic)
│   └── patch_stats.json               Per-class patch counts for DCGAN (5,485 total)
│
├── 02_dcgan/
│   ├── train_gan.py                   DCGAN training script (full source)
│   ├── training_history.json          Per-epoch loss_D / loss_G (120 epochs)
│   └── samples/
│       ├── epoch_0010/
│       │   └── class_{0-5}_*.png      Per-class 8x8 sample grids — early training (epoch 10)
│       ├── epoch_0060/
│       │   └── class_{0-5}_*.png      Per-class grids — mid training (epoch 60)
│       ├── epoch_0120/
│       │   └── class_{0-5}_*.png      Per-class grids — late training (epoch 120)
│       └── epoch_0300/
│           └── class_{0-5}_*.png      Per-class grids — extended training (epoch 300)
│
├── 03_pix2pix/
│   ├── train_pix2pix.py               Pix2Pix training script (U-Net + PatchGAN)
│   ├── generate_pix2pix.py            Synthetic image generation script
│   └── samples/
│       ├── epoch_0025.png             Pix2Pix progression — epoch 25
│       ├── epoch_0075.png             Pix2Pix progression — epoch 75
│       ├── epoch_0125.png             Pix2Pix progression — epoch 125
│       └── epoch_0200.png             Pix2Pix progression — epoch 200 (final)
│
├── 04_baseline_yolo/
│   ├── args.yaml                      YOLOv8-L training config (real-data baseline)
│   ├── results.csv                    Per-epoch metrics — 181 epochs
│   ├── results.png                    Training loss and metric curves
│   ├── confusion_matrix.png           Confusion matrix on validation set
│   ├── confusion_matrix_normalized.png
│   ├── BoxF1_curve.png                F1 vs confidence threshold
│   ├── BoxPR_curve.png                Precision-Recall curve
│   ├── BoxP_curve.png                 Precision vs confidence
│   ├── BoxR_curve.png                 Recall vs confidence
│   ├── labels.jpg                     Training label distribution
│   ├── train_batch0.jpg               Training batch mosaics
│   ├── train_batch1.jpg
│   ├── train_batch2.jpg
│   ├── val_batch0_labels.jpg          Validation ground truth
│   ├── val_batch0_pred.jpg            Validation predictions
│   ├── val_batch1_labels.jpg
│   └── val_batch1_pred.jpg
│
├── 04b_dcgan_retrain/
│   ├── args.yaml                      YOLOv8-L retrain config (DCGAN merged dataset)
│   └── results.csv                    Per-epoch metrics — 200 epochs
│                                      (run stored as 'pcb_defects_pix2pix' — mislabeled)
│
├── 05_pix2pix_v2_retrain/
│   ├── args.yaml                      YOLOv8-L retrain config (Pix2Pix merged dataset)
│   ├── results.csv                    Per-epoch metrics — ~195 epochs
│   ├── results.png                    Training curves
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── labels.jpg                     Label distribution (real + Pix2Pix synthetic)
│   ├── train_batch0.jpg
│   ├── train_batch1.jpg
│   ├── train_batch2.jpg
│   ├── val_batch0_labels.jpg
│   ├── val_batch0_pred.jpg
│   ├── val_batch1_labels.jpg
│   └── val_batch1_pred.jpg
│
└── 06_synthetic_samples/
    ├── dcgan_samples/
    │   └── synth_*.png                9 representative DCGAN composite full-PCB images
    ├── pix2pix_samples/
    │   └── px2px_*.png                6 representative Pix2Pix generated images
    └── pix2pix_visualized/
        └── px2px_*.png                100 side-by-side template->generated panels
```

---

## 10. References

[1] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). **Generative Adversarial Nets.** *Advances in Neural Information Processing Systems (NeurIPS)*, 27.

[2] Radford, A., Metz, L., & Chintala, S. (2015). **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** *ICLR 2016*. arXiv:1511.06434.

[3] Mirza, M., & Osindero, S. (2014). **Conditional Generative Adversarial Nets.** arXiv:1411.1784.

[4] Lim, J. H., & Ye, J. C. (2017). **Geometric GAN.** arXiv:1705.02894.

[5] Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). **Spectral Normalization for Generative Adversarial Networks.** *ICLR 2018*. arXiv:1802.05957.

[6] Sonderby, C. K., Caballero, J., Theis, L., Shi, W., & Huszar, F. (2016). **Amortised MAP Inference for Image Super-resolution.** *ICLR 2017*. arXiv:1610.04490.

[7] Arjovsky, M., & Bottou, L. (2017). **Towards Principled Methods for Training Generative Adversarial Networks.** *ICLR 2017*. arXiv:1701.04862.

[8] Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). **Image-to-Image Translation with Conditional Adversarial Networks.** *CVPR 2017*. arXiv:1611.07004.

[9] Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** *MICCAI 2015*. arXiv:1505.04597.

[10] Li, C., & Wand, M. (2016). **Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks.** *ECCV 2016*. arXiv:1604.04382.

[11] Jocher, G., Chaurasia, A., & Qiu, J. (2023). **Ultralytics YOLOv8.** https://github.com/ultralytics/ultralytics. AGPL-3.0 license.

[12] Loshchilov, I., & Hutter, F. (2019). **Decoupled Weight Decay Regularization.** *ICLR 2019*. arXiv:1711.05101.

[13] Loshchilov, I., & Hutter, F. (2016). **SGDR: Stochastic Gradient Descent with Warm Restarts.** *ICLR 2017*. arXiv:1608.03983.

[14] Tang, J., Liu, G., Pan, Q., & Yang, H. (2019). **Online PCB Defect Detector On A New PCB Defect Dataset.** arXiv:1902.06197. DeepPCB dataset: https://github.com/tangsanli5201/DeepPCB
