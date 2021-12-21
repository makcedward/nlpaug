import numpy as np

from nlpaug.util import Method
from nlpaug import Augmenter


class SpectrogramAugmenter(Augmenter):
    def __init__(self, action, zone=None, coverage=None, factor=None, name='Spectrogram_Aug', device='cpu', 
        verbose=0, stateless=True, silence=False):
        super().__init__(name=name, method=Method.SPECTROGRAM, action=action, aug_min=None, 
            aug_max=None, device=device, verbose=verbose)

        self.zone = zone
        self.coverage = coverage
        self.factor = factor
        self.stateless = stateless
        self.silence = silence

        if self.zone[0] < 0:
            raise ValueError('Lower bound of zone is smaller than {}.'.format(0) + 
                ' It should be larger than {}'.format(0))

        if self.zone[1] > 1:
            raise ValueError('Upper bound of zone is larger than {}.'.format(1) + 
                ' It should be smaller than {}'.format(1))

        if self.coverage < 0 or self.coverage > 1:
            raise ValueError('Coverage value should be between than 0 and 1 while ' +
                'input value is {}'.format(self.coverage))

    @classmethod
    def clean(cls, data):
        return data

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if np.array_equal(d, data):
                return True
        return False

    def get_random_factor(self, low=None, high=None, dtype='float'):
        lower_bound = self.factor[0] if low is None else low
        upper_bound = self.factor[1] if high is None else high
        if dtype == 'int':
            return np.random.randint(lower_bound, upper_bound)
        elif dtype == 'float':
            return np.random.uniform(lower_bound, upper_bound)
        else:
            return np.random.uniform(lower_bound, upper_bound)

    def get_augment_range_by_coverage(self, data):
        zone_start, zone_end = int(data.shape[1] * self.zone[0]), int(data.shape[1] * self.zone[1])
        zone_size = zone_end - zone_start

        target_size = int(zone_size * self.coverage)
        last_start = zone_start + int(zone_size * (1 - self.coverage))

        if zone_start == last_start:
            start_pos = zone_start
            end_pos = zone_end
        else:
            start_pos = np.random.randint(zone_start, last_start)
            end_pos = start_pos + target_size

        return start_pos, end_pos
