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
To run both the search and fine-tune steps of ODiMO a shell script is provided (e.g., `run_ic.sh`).

In the case of the Tiny ImageNet task the shell script takes 7 argument as input. The relevant one are detailed within `{__}`, the other can be leaved as fixed:
```
$ source run_ic.sh {regularization strenght} {architecture} 64 {regularization target} now search ft
```

where `{regularization strength}` is a float number that controls the balance between task loss and regularization loss. `{architecture}` is the architecture that we want to optimize with the tool. All the supported architectures are listed in [models directory](./models). E.g., for Tiny ImageNet task use as architecture `res18_pow2_diana_full`. `{regularization target}` represents the hw-related metric that we want to optimize, the supported values are `[power, latency, power-naive]`.

## License
ODiMO is released under Apache 2.0, see the LICENSE file in the root of this repository for details.