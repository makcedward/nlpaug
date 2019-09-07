try:
    import torch
    import torch.nn.functional as F
except:
    # No installation required if not using this function
    pass
import numpy as np


def validate_mode(mode):
    if mode not in ['larger', 'smaller']:
        raise ValueError('Mode should be either larger or smaller while {} is passed.'.format(mode))


def filter_proba(data, p, mode='larger', replace=None):
    """

    :param numpy data: Input data
    :param float p: Probability for filtering
    :param string mode: Either 'larger' or 'smaller'. If 'larger' is passed, only value larger than p will be kept
        (or not replaced)
    :param float replace: Default value is None. If value is provided, input data will be replaced by this value
        if data match criteria.
    :return: numpy Filtered result
    """
    validate_mode(mode)

    if replace:
        new_data = data.copy()
        if mode == 'larger':
            idxes = np.argwhere(data > p).flatten()
            replace_idxes = np.argwhere(data <= p).flatten()
        elif mode == 'smaller':
            idxes = np.argwhere(data < p).flatten()
            replace_idxes = np.argwhere(data >= p).flatten()
        new_data[replace_idxes] = replace

        return new_data, idxes

    if mode == 'larger':
        idxes = np.argwhere(data >= p).flatten()
    elif mode == 'smaller':
        idxes = np.argwhere(data <= p).flatten()

    return data[idxes], idxes


def filter_top_n(data, n, replace=None):
    """

    :param numpy data: Input data
    :param float n: Number of top element will be reserved (or not replaced)
    :param float replace: Default value is None. If value is provided, input data will be replaced by this value
        if data match criteria.
    :return: numpy Filtered result
    """
    if isinstance(data, np.ndarray):
        return filter_top_n_numpy(data, n, replace)
    if isinstance(data, torch.Tensor):
        return filter_top_n_pytorch(data, n, replace)
    raise ValueError("Only support numpy or pytorch's tensor while {} is provided".format(type(data)))


def filter_top_n_numpy(data, n, replace=None):
    # if n >= len(data):
    #     n = len(data)
        # return data, np.array([i for i, _ in enumerate(data)])

    idxes = np.argpartition(data, -n)[-n:]
    # print('idxes:', idxes)

    if replace:
        replace_idxes = np.argpartition(data, len(data)-n)[:len(data)-n]
        _data = data.copy()
        _data[replace_idxes] = replace
        return _data, idxes

    return data[tuple([idxes])], idxes


def filter_top_n_pytorch(data, n, replace=None):
    _data = data.clone()

    filtered_data, idxes = torch.topk(_data, n)

    if replace:
        _data[_data < filtered_data[-1]] = replace
        return _data, idxes

    return filtered_data, idxes
