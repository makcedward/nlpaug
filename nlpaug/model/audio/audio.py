import numpy as np


class Audio:
    def __init__(self, zone=(0.2, 0.8), coverage=1., factor=None, duration=None,
                 sampling_rate=None, stateless=True):
        self.zone = zone
        self.coverage = coverage
        self.factor = factor
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.stateless = stateless

        self.start_pos = None
        self.end_pos = None
        self.aug_data = None
        self.aug_factor = None

    @classmethod
    def pad(cls, data, noise):
        if len(data) - len(noise) == 0:
            start_pos = 0
        else:
            start_pos = np.random.randint(0, len(data) - len(noise))

        prefix_padding = np.array([0] * start_pos)
        suffix_padding = np.array([0] * (len(data) - len(noise) - start_pos))
        return np.append(np.append(prefix_padding, noise), suffix_padding)

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

    def manipulate(self, data):
        raise NotImplementedError
