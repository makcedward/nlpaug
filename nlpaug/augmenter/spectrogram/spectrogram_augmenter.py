import numpy as np

from nlpaug.util import Method
from nlpaug import Augmenter


class SpectrogramAugmenter(Augmenter):
    def __init__(self, action, name='Spectrogram_Aug', aug_min=1, aug_p=0.3, verbose=0):
        super(SpectrogramAugmenter, self).__init__(
            name=name, method=Method.SPECTROGRAM, action=action, aug_min=aug_min, verbose=verbose)
        self.aug_p = aug_p

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if np.array_equal(d, data):
                return True
        return False
