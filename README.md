

# **Paper:** KAT-ReID: Assessing the Viability of Kolmogorov–Arnold Transformers in Object Re-Identification


KAT-ReID is a Transformer-based ReID pipeline that keeps **self-attention** but replaces the **ViT MLP channel mixer** with a **Group-Rational Kolmogorov–Arnold Network (GR-KAN)** module, and retains ReID-specific components (side-information conditioning + a lightweight local token rearrangement branch). 

> Key finding: KAT-ReID’s gains concentrate in **occlusion-centric regimes** (Occluded-Duke), while it underperforms a strong ViT baseline on “cleaner” person/vehicle ReID benchmarks. 

---

## Why KAT-ReID?

Kolmogorov–Arnold Transformer (KAT) replaces MLP blocks with GR-KAN layers and has been reported to surpass vanilla ViTs on ImageNet classification (and other vision tasks), motivating the question: *does that representational advantage transfer to ReID?* 

ReID is a **fine-grained retrieval** problem under **occlusions and view shifts**, where “holistic cues” may collapse; KAT-ReID evaluates whether GR-KAN mixers help preserve discriminative identity evidence when only partial observations are available. 

---

## What’s in this repo?

Core structure (folders/scripts) includes:

* `train.py`, `test.py` (training + evaluation entrypoints)
* `configs/` (experiment configs)
* `dist_train.sh` (distributed training launcher)
* `katransformer.py` (KAT/GR-KAN backbone code)
* supporting modules: `model/`, `loss/`, `solver/`, `processor/`, `utils/`, `datasets/`
  ([GitHub][1])

---

## Method overview

### 1) Backbone: MLP → GR-KAN (drop-in mixer)

Each transformer block keeps MHSA + residuals, but replaces the 2-layer GELU MLP with a two-stage GR-KAN mixer using group-wise learnable rational activations and variance-preserving initialization. 

### 2) ReID-specific components kept

* **Side-information conditioning (SIE):** adds a learnable bias encoding camera and/or view IDs, scaled by coefficient ( \lambda_{SIE} ). 
* **Local token rearrangement branch:** cyclic shift + regroup spatial tokens into **K local descriptors**; default **K=4**, concatenated with the global descriptor at inference. 

---

## Reported results (single-query mAP / CMC)

Table below reproduces the paper’s Table II (higher is better). 

| Dataset       |              Method |      mAP |      R-1 |  R-5 | R-10 |
| ------------- | ------------------: | -------: | -------: | ---: | ---: |
| Market-1501   |     KAT-ReID (ours) |     78.8 |     91.1 | 96.9 | 98.2 |
| Market-1501   | TransReID (ICCV’21) |     88.9 |     95.2 |    – |    – |
| MSMT17        |     KAT-ReID (ours) |     38.8 |     61.5 | 76.6 | 82.2 |
| MSMT17        | TransReID (ICCV’21) |     67.4 |     85.3 |    – |    – |
| Occluded-Duke |     KAT-ReID (ours) | **69.6** | **83.7** | 91.8 | 94.5 |
| Occluded-Duke | TransReID (ICCV’21) |     59.2 |     66.4 |    – |    – |
| VeRi-776      |     KAT-ReID (ours) |     59.5 |     88.0 | 95.8 | 98.0 |
| VeRi-776      | TransReID (ICCV’21) |     80.5 |     96.8 |    – |    – |

**Occluded-Duke improvement:** +10.4 mAP and +17.3 Rank-1 over the ViT baseline. 

---

## Datasets

The paper evaluates four benchmarks (official splits, single-query evaluation): Market-1501, MSMT17, Occluded-Duke, VeRi-776. 

Dataset stats (paper Table I): 

* **MSMT17:** 4,101 IDs, 126,441 images, 15 cameras
* **Market-1501:** 1,501 IDs, 32,668 images, 6 cameras
* **Occluded-Duke:** 1,404 IDs, 36,441 images, 8 cameras
* **VeRi-776:** 776 IDs, 49,357 images, 20 cameras, 8 views

> **Note:** Please follow each dataset’s license/terms from its official source.

---

## Installation

1. **Clone**

```bash
git clone https://github.com/muhammadumair894/KAT-ReID.git
cd KAT-ReID
```

2. **Create env + install dependencies**

```bash
conda create -n katreid python=3.8 -y
conda activate katreid
pip install -r requirements.txt
```

A `requirements.txt` is provided at repo root. 

---

## Pretrained initialization (important)

Experiments initialize from **ImageNet pretraining**; for KAT backbones, linear layers inherit ViT MLP weights while GR-KAN activations are initialized to preserve variance. 
The paper explicitly notes to **provide the path to the ImageNet checkpoint** in your run/config. 

---

## How to run

This repo ships:

* `train.py` for training 
* `test.py` for evaluation 
* `cmd4test-train.md` containing example commands (recommended starting point). 

### 1) Configure your experiment

Pick/edit a config under `configs/` (dataset name, paths, model settings, checkpoint path). `configs/` exists in the repo root. 

### 2) Train (single GPU)

```bash
python train.py --config_file <PATH_TO_CONFIG>
```

Training uses **mixed precision** in the paper setup. 


### 3) Test / Evaluate

```bash
python test.py --config_file <PATH_TO_CONFIG> TEST.WEIGHT <PATH_TO_CHECKPOINT>
```

---

## Reproducing the paper setup (key hyperparameters)

The following settings are stated in the paper (match these if you want comparable numbers):

### Input & tokenization

* Resize images to **256×128**
* Patch size **16×16** with **overlap stride (12, 12)**
* Add learnable 2D positional embedding + prepend `[CLS]`
* If camera IDs are available: side-information bias with ( \lambda_{SIE}=3.0 ) (camera only; view off by default)


### Backbone & regularization

* Pre-norm transformer blocks; stochastic depth (DropPath) linearly increases to **0.1**


### Local branch

* Enabled at last stage; produces **K=4** local descriptors (default)


### Optimization

* Optimizer: **SGD**, base LR **0.008**, momentum **0.9**, weight decay **1e−4**

* Losses: ID softmax CE (label smoothing off), + triplet (soft-margin; hinge margin 0.3 if used), both weight 1.0

* Augmentations: random flip p=0.5, random erasing p=0.5; padding 10; normalize mean/std (0.5,0.5,0.5)

* Sampler / batch: softmax_triplet sampler, global batch 64 = **P=16 IDs × K=4 images**


### Evaluation protocol

* Evaluate every **120 epochs**
* Test batch size **256**
* **Re-ranking: False**
* Retrieval uses `NECK_FEAT=before` with ℓ2 normalization enabled
* Seed fixed to **1234**


---

## Notes on behavior 

* KAT-ReID’s advantage is strongest when **occlusion / partial observability dominates**; on globally-biased benchmarks it may lag unless further stabilization/pretraining calibration is applied. 
* Identified directions to close gaps include longer warmup, stronger regularization on activation parameters, modest label smoothing, calibrated branch losses, and slightly increased patch overlap. 

---

## Citation

If you use this code, please cite the paper and the key baselines.

```bibtex
@inproceedings{umair2025katreid,
  title     = {KAT-ReID: Assessing the Viability of Kolmogorov--Arnold Transformers in Object Re-Identification},
  author    = {Umair, Muhammad and Jun, Zhou and Musaddiq, Muhammad Hammad and Muhammad, Ahmad},
  year      = {2025},
  note      = {ICCC-2025}
}
```

(Recommended additional citations from the paper’s references:)

* ViT 
* KAN 
* KAT (arXiv:2409.10594) 
* TransReID baseline (ICCV’21) appears in Table II 

---

## Contact

* Muhammad Umair — [muhammadumair894@gmail.com](mailto:muhammadumair894@gmail.com) 

---
