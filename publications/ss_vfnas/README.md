# Cross-silo Federated Neural Architecture Search for Heterogeneous and Cooperative Systems

## Overview
We present Self-supervised Vertical Federated Neural Architecture Search (SS-VFNAS) for automating FL where participants hold heterogeneous data and devices, a common cross-silo scenario. SS-VFNAS simultaneously optimized all parties' model architecture and parameters for the best global performance under a vertical FL (VFL) framework using only a small set of aligned and labeled data whereas preserving each party's local optimal model architecture under a self-supervised NAS framework using unaligned and unlabeled local data.
## Dataset
* modelnet40v1png: [maxwell.cs.umass.edu/mvcnn-data](http://maxwell.cs.umass.edu/mvcnn-data/)
* chexpert: [stanfordmlgroup.github.io](https://stanfordmlgroup.github.io/competitions/chexpert/)

## Getting Started
### Install dependencies
## Requirements

- python3
- pytorch
- graphviz
    - First install using `apt install` and then `pip install`.
- numpy
- tensorboardX

## Run example

Adjust the batch size if out of memory (OOM) occurs. It dependes on your gpu memory size and genotype.

- Search

```shell
python train_search_k_party.py --data data/modelnet40v1png --name test 
```

- Search with self-supervised
```shell
python train_search_k_party_moco.py --data data/modelnet40v1png --name test
```

- Augment

```shell
python train_darts_k_party.py --data data/modelnet40v1png --name test --genotypes_file path_to_genotypes
```

- Train manual network
```shell
python train_manual_k_party.py --data data/modelnet40v1png --name test --layers 18
```