import numpy as np

from nlpaug.util import Method
from nlpaug import Augmenter


class AudioAugmenter(Augmenter):
    def __init__(self, action, zone=None, coverage=None, factor=None, duration=None, name='Audio_Aug', 
        device='cpu', verbose=0, stateless=True):
        super(AudioAugmenter, self).__init__(
            name=name, method=Method.AUDIO, action=action, aug_min=None, aug_max=None, device=device, 
            verbose=verbose)

        self.zone = zone
        self.coverage = coverage
        self.factor = factor
        self.duration = duration
        self.stateless = stateless

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
        lower_bound = low if low else self.factor[0]
        upper_bound = high if high else self.factor[1]
        if dtype == 'int':
            return np.random.randint(lower_bound, upper_bound)
        elif dtype == 'float':
            return np.random.uniform(lower_bound, upper_bound)
        
        return np.random.uniform(lower_bound, upper_bound)

    def get_augmentation_segment_size(self, data):
        return int(len(data) * (self.zone[1] - self.zone[0]) * self.coverage)

    def get_augment_range_by_coverage(self, data):
        zone_start, zone_end = int(len(data) * self.zone[0]), int(len(data) * self.zone[1])
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

    def get_augment_range_by_duration(self, data):
        zone_start, zone_end = int(len(data) * self.zone[0]), int(len(data) * self.zone[1])
        zone_size = zone_end - zone_start

        target_size = int(self.sampling_rate * self.duration)

        if target_size >= zone_size:
            start_pos = zone_start
            end_pos = zone_end
        else:
            last_start = zone_start + zone_size - target_size
            start_pos = np.random.randint(zone_start, last_start)
            end_pos = start_pos + target_size

        return start_pos, end_pos