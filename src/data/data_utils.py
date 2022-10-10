import os
import random

import torch
import warnings
import torch.utils.data


def get_data_path(data_root, dataname: str):
    if data_root is None:
        return None  # using cached data
    if os.path.isdir(os.path.join(data_root, dataname)):
        return os.path.join(data_root, dataname)
    if os.path.isdir(os.path.join(data_root, dataname.lower())):
        return os.path.join(data_root, dataname.lower())
    if os.path.isdir(os.path.join(data_root, dataname.upper())):
        return os.path.join(data_root, dataname.upper())
    raise ValueError(f'Error - Could not locate data {os.path.join(data_root, dataname)}')


def get_data(data_root, dataname, image_size,
             split, regime='neutral', subset=None, use_cache=False, save_cache=False,
             batch_size=None, drop_last=True, num_workers=0, ratio=None, shuffle=True, flip=False, permute=False):

    # Load real dataset
    if 'PGM' in dataname:
        from .pgm_dataset import PGMDataset
        dataset = PGMDataset(get_data_path(data_root, dataname), None,
                             use_cache=use_cache, save_cache=save_cache,
                             split=split, regime=regime, subset=subset,
                             image_size=image_size, transform=None, flip=flip, permute=permute)

    elif 'RAVEN' in dataname:
        from .raven_dataset import RAVENDataset
        dataset = RAVENDataset(get_data_path(data_root, dataname), None,
                               use_cache=use_cache, save_cache=save_cache,
                               split=split, subset=subset,
                               image_size=image_size, transform=None, flip=flip, permute=permute)

    else:
        raise ValueError(dataname)

    # Reduce dataset to a smaller subset, nice for debugging
    if ratio is not None:
        old_len = len(dataset)
        indices = list(range(old_len))
        random.shuffle(indices)
        dataset = torch.utils.data.Subset(dataset, indices[:int(max(old_len * ratio, 5 * batch_size))])
        warnings.warn(f'Reducing dataset size from {old_len} to {len(dataset)}')

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             drop_last=drop_last,
                                             num_workers=num_workers)

    return dataloader