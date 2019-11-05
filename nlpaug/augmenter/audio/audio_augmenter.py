import numpy as np

from nlpaug.util import Method
from nlpaug import Augmenter


class AudioAugmenter(Augmenter):
    def __init__(self, action, name='Audio_Aug', device='cpu', verbose=0):
        super(AudioAugmenter, self).__init__(
            name=name, method=Method.AUDIO, action=action, aug_min=None, aug_max=None, device=device, verbose=verbose)

    def substitute(self, data):
        return self.model.manipulate(data)

    @classmethod
    def clean(cls, data):
        return data

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if np.array_equal(d, data):
                return True
        return False
