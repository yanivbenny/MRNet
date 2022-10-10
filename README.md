# MRNet - Multi-scale Reasoning Network
Official repository for:

Yaniv Benny, Niv Pekar, Lior Wolf. [**"Scale-Localized Abstract Reasoning"**](https://openaccess.thecvf.com/content/CVPR2021/papers/Benny_Scale-Localized_Abstract_Reasoning_CVPR_2021_paper.pdf). CVPR 2021.

![architecture](images/architecture.png)


## Requirements
* python 3.6
* NVIDIA GPU with CUDA 10.0+ capability
* tqdm, PyYaml
* numpy, scipy, matplotlib, scikit-image
* torch==1.7.1, torchvision==0.8.2


## Data
* [PGM](https://github.com/deepmind/abstract-reasoning-matrices)
* [RAVEN](https://github.com/WellyZhang/RAVEN)
* [RAVEN-FAIR](https://github.com/yanivbenny/RAVEN_FAIR) (Our new version of RAVEN)


## Code
Optional:
* To speedup training, try running `save_cache.py` in advance.  \
This script will basically save the dataset after resizing all the images from 160x160 to 80x80 in a separate location so that this won't have to be done during runtime. \
This will reduce a lot of CPU utilization and disk reads during training. \
`$ python save_cache.py --data_dir <PATH-TO-DATASETS --dataset <DATASET>` \
If you have done this step, add `--use_cache` to the training command.


To reproduce the results, run:
1. First training \
`$ CUDA_VISIBLE_DEVICES=0 python train.py --dataset <DATASET> --data_dir <PATH-TO-DATASETS> --wd <WD> --multihead`
2. When first training is done \
`$ CUDA_VISIBLE_DEVICES=0 python train.py --dataset <DATASET> --data_dir <PATH-TO-DATASETS> --wd <WD> --recovery --multihead --multihead_mode eprob`
* For PGM use WD=0. For RAVEN-like use WD=1e-5. 

To run test only, add `--recovery --test` to the command.

## Pretrained models 
Download the pretrained models for PGM and RAVEN-FAIR [here](https://drive.google.com/drive/folders/1ss1ZSSZ3SOH7O8vrqUw4jeAkxYuiYmTx?usp=sharing). \
Put the model inside a folder `<EXP-DIR>/<EXP-NAME>/save` and specify `--exp_dir <EXP-DIR> --exp_name <EXP-NAME> --recovery --test`

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
