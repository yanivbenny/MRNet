# MRNet - Multi-scale Reasoning Network
Official repository for:

Yaniv Benny, Niv Pekar, Lior Wolf. **"Scale-Localized Abstract Reasoning"**. CVPR 2021.

[paper](https://github.com/yanivbenny/MRNet).

![architecture](images/architecture.png)


## Requirements
* python 3.6
* NVIDIA GPU with CUDA 10.0+ capability
* numpy, scipy, matplotlib
* torch==1.4.0
* torchvision==0.5.0
* scikit-image


## Data
* [PGM](https://github.com/deepmind/abstract-reasoning-matrices)
* [RAVEN](https://github.com/WellyZhang/RAVEN)
* [RAVEN-FAIR](https://github.com/yanivbenny/RAVEN_FAIR) (Our new version of RAVEN)


## Code
To reproduce the results, run:
1. First training \
`$ CUDA_VISIBLE_DEVICES=0 python train.py --dataset <DATASET> --path <PATH-TO-DATASETS> --wd <WD> --multihead`
2. When first training is done \
`$ CUDA_VISIBLE_DEVICES=0 python train.py --dataset <DATASET> --path <PATH-TO-DATASETS> --wd <WD> --recovery --multihead --multihead_mode eprob`
* For PGM use WD=1e-6. For RAVEN-like use WD=1e-5.

## Pretrained models 
Download the pretrained models for PGM and RAVEN-FAIR [here](https://drive.google.com/drive/folders/1ss1ZSSZ3SOH7O8vrqUw4jeAkxYuiYmTx?usp=sharing).

## Citation
We thank you for showing interest in our work. 
If our work was beneficial for you, please consider citing us using:

```
@inproceedings{benny2021scale,
  title={Scale-localized abstract reasoning},
  author={Benny, Yaniv and Pekar, Niv and Wolf, Lior},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12557--12565},
  year={2021}
}
```

If you have any question, please feel free to contact us.
