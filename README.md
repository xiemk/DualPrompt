# DualPrompt

Official code for the ICLR 2026 paper **"Unlocking the Power of Co-Occurrence in CLIP: A DualPrompt-Driven Method for Training-Free Zero-Shot Multi-Label Classification"**.

## Overview

DualPrompt is a **training-free zero-shot multi-label classification** method built on CLIP. It improves label prediction by injecting class co-occurrence information into text prompts through a dual-prompt design:

- **Discriminative prompts** for class-specific recognition
- **Correlation-aware prompts** for modeling label co-occurrence

## Features

- Training-free zero-shot multi-label classification with CLIP
- Dual-prompt design: discriminative prompt + correlation-aware prompt
- Multiple co-occurrence sources:
  - `data`: estimated from a subset of training labels
  - `chatgpt`: external semantic co-occurrence priors stored in JSON files
  - `prior`: full training-set co-occurrence prior
- Metrics include **mAP, CP, CR, CF1, OP, OR, OF1**

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Example on COCO2014 with CLIP ViT-B/16:

```bash
python run.py --data_name coco2014 --data_dir ./data --model_name ViT-B/16 --coo_derive data --sample_ratio 0.01 --coo_mode both --coo_max_n 5 --lam 0.5 \
```

## Main Arguments

- `--data_name`: dataset name, one of `coco2014`, `vg256`, `objects365`
- `--data_dir`: root path containing dataset folders
- `--model_name`: CLIP backbone
- `--output`: output root directory
- `--batch_size`: batch size for evaluation
- `--workers`: number of dataloader workers
- `--lam`: weight of the correlation-aware branch in the final dual-logit fusion
- `--coo_derive`: co-occurrence source, one of `data`, `chatgpt`, `prior`
- `--coo_mode`: co-occurrence construction mode, one of `top`, `bottom`, `both`
- `--coo_thre`: threshold for keeping co-occurring classes
- `--coo_max_n`: maximum number of co-occurring classes per target class
- `--sample_ratio`: ratio of training labels used to estimate co-occurrence
- `--seed`: random seed

## Outputs

Results are saved under the directory specified by `--output`. The final release version keeps only the **dual-logit** outputs and summary metrics.

Typical files include:

- `config.json`: full runtime configuration
- `log.txt`: evaluation log
- `coo_dict.json`: generated co-occurrence dictionary when `--coo_derive data`
- `dual_logit.npy`: final logits of the dual-prompt method
- `dual_prob.npy`: final normalized prediction scores used for evaluation
- `APs.npy`: per-class average precision values
- `summary_metrics.json`: summary metrics including `mAP`, `CP`, `CR`, `CF1`, `OP`, `OR`, and `OF1`
- `mAP.npy`, `CF1.npy`, `OF1.npy`: key scalar metrics saved separately


## Citation

```bibtex
@inproceedings{
xie2026unlocking,
title={Unlocking the Power of Co-Occurrence in CLIP: A DualPrompt-Driven Method for Training-Free Zero-Shot Multi-Label Classification},
author={Ming-Kun Xie and Zhiqiang Kou and Zhongnian Li and Gang Niu and Masashi Sugiyama},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=QGXVZ0OPLy}
}
```
