import torch
import numpy as np
import numbers


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
    if isinstance(arr, numbers.Number):
        t = torch.ones(1, 1)*arr
    elif type(arr) is list or type(arr) is tuple:
        t = torch.Tensor(arr).view(len(arr), 1)
    else:
        t = torch.from_numpy(np.array(arr))
    return t.type(dtype)

