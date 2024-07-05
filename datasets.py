
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-6

def read_dataset(data_root, data_name, flag=None, conditioning_parameters=None):
    """ Read a dataset """
    if flag is not None:
        data_name = flag + '_' + data_name
    data = np.load(os.path.join(data_root, f'{data_name}.npz'))
    cond_table = pd.read_csv(os.path.join(data_root, f'{data_name}_cond.csv'))
    if conditioning_parameters is not None:
        cond_table = cond_table[conditioning_parameters]
    return data, cond_table

def read_preprocess_dataset(
    data_root, data_name, flag=None, conditioning_parameters=None,
    norm_dict=None, norm_data=True, invert_mask=True, return_numpy=False,
    dtype=torch.float32):
    """ Read and preprocess a dataset

    Parameters
    ----------
    data_root : str
        Path to the data root directory
    data_name : str
        Name of the dataset
    flag : str, optional
        Flag to append to the dataset name
    conditioning_parameters : list of str, optional
        List of conditioning parameters to use
    norm_dict : dict, optional
        Dictionary containing normalization parameters
    norm_data : bool, optional
        Whether to normalize the data. Default is True.
    invert_mask : bool, optional
        Whether to invert the mask. This is necessary because the mask convention
        is different between torch and jax. Default is True.
    return_numpy : bool, optional
        Whether to return numpy arrays instead of torch tensors. Default is False.
    dtype : torch.dtype, optional
        Data type of the tensors. Default is torch.float32.
    """
    # read and expand the dataset
    data, cond_table = read_dataset(
        data_root, data_name, flag, conditioning_parameters)
    x = data['features']
    mask = data['mask']
    cond = cond_table.values

    # normalize the data
    if norm_data:
        if norm_dict is None:
            mask_bool = mask.astype(bool)
            x_mean = np.mean(x, axis=(0, 1), where=mask_bool[..., None])
            x_std = np.std(x, axis=(0, 1), where=mask_bool[..., None])
            cond_mean = np.mean(cond, axis=0)
            cond_std = np.std(cond, axis=0)
            norm_dict = {
                'x_mean': x_mean,
                'x_std': x_std,
                'cond_mean': cond_mean,
                'cond_std': cond_std
            }
        else:
            x_mean = norm_dict['x_mean']
            x_std = norm_dict['x_std']
            cond_mean = norm_dict['cond_mean']
            cond_std = norm_dict['cond_std']
        x = (x - x_mean + EPS) / (x_std + EPS)
        cond = (cond - cond_mean + EPS) / (cond_std + EPS)
    else:
        norm_dict = None

    if return_numpy:
        if invert_mask:
            mask = ~mask
        return x, cond, mask, norm_dict
    else:
        x = torch.tensor(x, dtype=dtype)
        cond = torch.tensor(cond, dtype=dtype)
        mask = torch.tensor(mask, dtype=torch.bool)
        if invert_mask:
            mask = ~mask
        return x, cond, mask, norm_dict

def create_dataloader(data, **kwargs):
    """ Create a PyTorch DataLoader """
    dataset = TensorDataset(*data)
    return DataLoader(dataset, **kwargs)
