try:
    import torch
    import torch.nn.functional as F
except ImportError:
    # No installation required if not using this function
    pass
import numpy as np


def filter_proba(data, p, replace=None, above=True):
    """

    :param numpy data: Input data
    :param float p: Probability for filtering (or be replaced)
    :param float replace: Default value is None. If value is provided, input data will be replaced by this value
        if data match criteria.
    :param bool above: If True is passed, only value larger than p will be kept (or not replaced)
    :return: numpy Filtered result
    """

    if replace:
        new_data = data.copy()
        if above:
            idxes = np.argwhere(data > p).flatten()
            replace_idxes = np.argwhere(data <= p).flatten()
        else:
            idxes = np.argwhere(data < p).flatten()
            replace_idxes = np.argwhere(data >= p).flatten()
        new_data[replace_idxes] = replace

        return new_data, idxes

    if above:
        idxes = np.argwhere(data >= p).flatten()
    else:
        idxes = np.argwhere(data <= p).flatten()

    return data[idxes], idxes


def filter_top_k(data, k, replace=None, ascending=False):
    """

    :param numpy/tensor data: Input data
    :param float k: Number of top element will be reserved (or not replaced)
    :param float replace: Default value is None. If value is provided, input data will be replaced by this value
        if data match criteria.
    :param bool ascending: Return ascending order or descending order. Sorting will be executed if replace is None.
    :return: numpy/tensor Filtered result
    """
    if isinstance(data, np.ndarray):
        return filter_top_k_numpy(data, k, replace, ascending)
    if isinstance(data, torch.Tensor):
        return filter_top_k_pytorch(data, k, replace, ascending)
    raise ValueError("Only support numpy or pytorch's tensor while {} is provided".format(type(data)))


def filter_top_k_numpy(data, k, replace=None, ascending=False):
    idxes = np.argpartition(data, -k)[-k:]

    if replace:
        replace_idxes = np.argpartition(data, len(data)-k)[:len(data)-k]
        _data = data.copy()
        _data[replace_idxes] = replace
        return _data, idxes

    if not ascending:
        idxes = idxes[::-1]

    return data[tuple([idxes])], idxes


def filter_top_k_pytorch(data, k, replace=None, ascending=False):
    _data = data.clone()

    filtered_data, idxes = torch.topk(_data, k)

    if replace:
        _data[_data < filtered_data[-1]] = replace
        return _data, idxes

    if ascending:
        return torch.flip(filtered_data, (0, )), torch.flip(idxes, (0, ))
    else:
        return filtered_data, idxes


# Source: http://arxiv.org/abs/1904.09751
def nucleus_sampling(data, p, replace=0, ascending=False, above=True):
    """

    :param tensor data: Input data
    :param float p: Probability for filtering (or be replaced)
    :param float replace: Default value is 0. If value is provided, input data will be replaced by this value
        if data match criteria.
    :param bool ascending: Return ascending order or descending order. Sorting will be executed if replace is None.
    :param bool above: If True is passed, only value smaller than p will be kept (or not replaced)
    :return: tensor Filtered result
    """
    sorted_data, sorted_indices = torch.sort(data, descending=not ascending)
    cum_probas = torch.cumsum(F.softmax(sorted_data, dim=-1), dim=-1)

    if replace is None:
        if above:
            replace_idxes = cum_probas < p
        else:
            replace_idxes = cum_probas > p
        idxes = sorted_indices[replace_idxes]
    else:
        if above:
            replace_idxes = cum_probas > p
        else:
            replace_idxes = cum_probas < p
        idxes = sorted_indices[~replace_idxes]
    if replace is None:
        sorted_data = sorted_data[replace_idxes]
    else:
        sorted_data[replace_idxes] = replace

    return sorted_data, idxes
