# Adversarial Invariant Learning
This repository contains code for the paper:

```latex
@article{AdversarialInvariantLearning,
    title={AdversarialInvariantLearning},
    author={Nanyang Ye, Jingxuan Tang, Xiao-Yun Zhou, Huayu Deng, Qianxiao Li, Zhenguo Li, Guang-Zhong Yang, Zhanxing Zhu},
    year={2021}
}
```

## MNIST and PSST experiments

The code used in the experiments are located in `./domainbed`, which is a fork of the [DomainBed](https://github.com/facebookresearch/DomainBed) repository(Licensed under the MIT license). Documentation can be found at the original repository. 

AIL, the proposed method in the paper, is implemented as `AIL`(for different MNIST datasets) in `algorithms.py`. Hyperparameter choices are in `hparams_registry.py`. A training launcher using multiple GPUs is in `command_launchers.py`.  Additional datasets(Colored KMNIST, Colored Fashion MNIST, Punctuated SST) can be used by manually downloading these datasets and specifying the dataset names(`CKMNIST` , `CFMNIST`, `PSST`) when launching. When training on non-MNIST datasets such as the Punctuated SST, the specifications of the VAE layers need to be manually changed in `vae.py`.

For example, the following command launches a training and model selection task of AIL and IRM algorithms, on the ColoredMNIST and Colored KMNIST datasets.

```
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher local\
       --algorithms IRM AIL\
       --datasets ColoredMNIST CKMNIST\
       --n_hparams 20\
       --n_trials 3
```

We provide the detail of selected model in one of the random trials for each of the ColoredMNIST and PSST dataset. They are located in the `*-Sample-Output` directory. The outputs contain hyper-parameter choices(at the beginning of `out.txt`), the model file and the accuracy in the training process.
