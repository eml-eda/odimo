Copyright (C) 2023 Politecnico di Torino, Italy. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Matteo Risso, Alessio Burrello, Giuseppe Maria Sarda, Luca Benini, Enrico Macii, Massimo Poncino, Marian Verhelst, Daniele Jahier Pagliari

# Precision-aware Latency and Energy Balancing on Multi-Accelerator Platforms for DNN Inference

## Reference
If you use ODiMO in your experiments, please make sure to cite our paper:
```
@misc{risso2023precisionaware,
      title={Precision-aware Latency and Energy Balancing on Multi-Accelerator Platforms for DNN Inference}, 
      author={Matteo Risso and Alessio Burrello and Giuseppe Maria Sarda and Luca Benini and Enrico Macii and Massimo Poncino and Marian Verhelst and Daniele Jahier Pagliari},
      year={2023},
      eprint={2306.05060},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Install
To install the latest release:
```
$ git clone https://github.com/eml-eda/odimo
$ cd odimo
$ python setup.py install
```

## Datasets
The current version include examples based upon the following datasets:
- [CIFAR10](./image_classification/cifar10/)
- [Tiny ImageNet](./image_classification/tiny-imagenet/)
- [Visual Wake Words](./visual_wake_words)

## How to run
Visit the directory with the task that you want to run (e.g., `image_classification/tiny-imagenet`).


## License
ODiMO is released under Apache 2.0, see the LICENSE file in the root of this repository for details.