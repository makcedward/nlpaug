import numpy as np

from nlpaug.util import Method
from nlpaug import Augmenter


class SpectrogramAugmenter(Augmenter):
    def __init__(self, action, name='Spectrogram_Aug', device='cpu', verbose=0):
        super(SpectrogramAugmenter, self).__init__(
            name=name, method=Method.SPECTROGRAM, action=action, aug_min=None, aug_max=None, device=device,
            verbose=verbose)

    @classmethod
    def clean(cls, data):
        return data

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if np.array_equal(d, data):
                return True
        return False
