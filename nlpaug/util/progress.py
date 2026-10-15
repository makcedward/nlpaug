import warnings

__all__ = ['optional_progress_bar']

_missing_tqdm_warned = False


def optional_progress_bar(iterable, enable=False, desc=None):
    """Wrap an iterable with tqdm when requested and available.

    :param iterable: Iterable to (optionally) wrap.
    :param bool enable: If False (default), return the iterable unchanged.
    :param str desc: Description shown next to the progress bar.
    :return: tqdm-wrapped iterable or the original iterable.

    Missing tqdm is not an error: a one-time warning is emitted and the
    original iterable is returned. Install it with `pip install tqdm` to
    enable progress bars (see issue #343).
    """
    if not enable:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        global _missing_tqdm_warned
        if not _missing_tqdm_warned:
            warnings.warn(
                'progress_bar=True requires tqdm (`pip install tqdm`). '
                'Continuing without a progress bar.',
                UserWarning,
                stacklevel=3,
            )
            _missing_tqdm_warned = True
        return iterable
    try:
        total = len(iterable)
    except TypeError:
        total = None
    return tqdm(iterable, total=total, desc=desc, leave=False)
