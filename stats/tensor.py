import torch
import numpy as np


def tensor(arr, dtype=torch.cuda.DoubleTensor):
    """
    Converts a float or an (numpy) array to a torch tensor.
    Parameters
    ----------
    arr :   float or list or ndarray
            Scalar or array of floats
    dtype : torch.dtype

    Returns
    -------
    Torch Tensor
    """
    if type(arr) is float or type(arr) is int:
        t = torch.ones(1)*arr
    elif type(arr) is list or type(arr) is tuple:
        t = torch.Tensor(arr)
    else:
        t = torch.from_numpy(np.array(arr))
    return t.type(dtype)

