import numpy as np

from nlpaug.model.audio import Audio


class Loudness(Audio):
    def __init__(self, zone=(0.2, 0.8), coverage=1., factor=(0.5, 2), stateless=True):
        """
        :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
            augmentation
        will be applied in first 20% and last 20% of whole audio.
        :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
            operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
            zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
            augmented.
        :param factor: Input data volume will be increased (decreased). Augmented value will be picked
            within the range of this tuple value. If volume will be reduced if value is between 0 and 1.
        """
        super().__init__(zone=zone, coverage=coverage, factor=factor, stateless=stateless)

    def get_loudness_level(self):
        return np.random.uniform(self.factor[0], self.factor[1])

    def manipulate(self, data):
        loudness_level = self.get_loudness_level()
        start_pos, end_pos = self.get_augment_range_by_coverage(data)
        aug_data = data[start_pos:end_pos] * loudness_level

        if not self.stateless:
            self.aug_factor = loudness_level
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.aug_data = aug_data

        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0)
