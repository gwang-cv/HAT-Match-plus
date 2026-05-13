# HAT-Match++

PyTorch code release for **HAT-Match++: Geometry-Explicit Hybrid Attention for Two-View Correspondence Pruning**.

This repository accompanies a manuscript that is currently under review.

## Overview

HAT-Match++ is a two-stage correspondence pruning network for robust two-view geometry estimation. The released implementation contains the core training and evaluation pipeline described in the manuscript:

- Stage-wise progressive pruning and score-conditioned refinement
- Enhanced Hybrid Attention Block with VCA, SA+, CSCA, and MCA
- Confidence-aware multi-hop graph attention
- Differentiable weighted eight-point estimation
- Relative-pose training and evaluation on precomputed correspondence dumps

## Repository Layout

```text
.
|-- config.py
|-- data.py
|-- evaluation.py
|-- hatmatchplus.py
|-- logger.py
|-- loss.py
|-- main.py
|-- test.py
|-- train.py
|-- transformations.py
|-- utils.py
`-- warmupMultiStepLR.py
```

## Requirements

The code was organized around the PyTorch-based training stack used by the manuscript release.

```bash
pip install -r requirements.txt
```

Core dependencies are listed in `requirements.txt`.

## Data

Please follow the data preparation protocol used by OANet. First download the YFCC100M dataset. Download the SUN3D testing and training datasets as needed. Then generate the correspondence dumps for YFCC100M and SUN3D before training or evaluation.

HAT-Match++ uses the same style of precomputed HDF5 correspondence files in its training and testing pipeline.

## Training

Example YFCC-SIFT training command:

```bash
python main.py \
  --run_mode train \
  --data_tr /path/to/yfcc-sift-2000-train.hdf5 \
  --data_va /path/to/yfcc-sift-2000-val.hdf5 \
  --data_te /path/to/yfcc-sift-2000-test.hdf5 \
  --log_base ./model/yfcc_sift_hatmatchplus \
  --train_batch_size 32
```

## Evaluation

Run evaluation from a saved model directory:

```bash
python main.py \
  --run_mode test \
  --data_te /path/to/yfcc-sift-2000-test.hdf5 \
  --model_path ./model/yfcc_sift_hatmatchplus/train \
  --res_path ./model/yfcc_sift_hatmatchplus/test \
  --use_ransac false
```

Set `--use_ransac true` to evaluate the robust-verification branch used in the original testing workflow.

## Acknowledgement

This project follows the broader learning-based correspondence pruning line represented by PointCN, OANet, and HAT-Match.
