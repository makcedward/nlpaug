import numpy as np

from nlpaug.model.audio import Audio


class Mask(Audio):
    def __init__(self, sampling_rate=None, zone=(0.2, 0.8), coverage=1., duration=None,
                 mask_with_noise=True, stateless=True):
        """
        :param int sampling_rate: Sampling rate of input audio. Mandatory if duration is provided.
        :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
            augmentation
        will be applied in first 20% and last 20% of whole audio.
        :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
            operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
            zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
            augmented.
        :param float duration: Duration of augmentation (in second). Default value is None. If value is provided. `coverage`
            value will be ignored.
        :param bool mask_with_noise: If it is True, targeting area will be replaced by noise. Otherwise, it will be
                replaced by 0.
        """
        super().__init__(zone=zone, coverage=coverage, duration=duration, sampling_rate=sampling_rate,
                         stateless=stateless)
        self.mask_with_noise = mask_with_noise

    def manipulate(self, data):
        start_pos, end_pos = self.get_augment_range_by_coverage(data)

        aug_data = None
        if self.mask_with_noise:
            aug_data = np.random.randn(end_pos - start_pos)
        else:
            aug_data = np.zeros(end_pos - start_pos)

        if not self.stateless:
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.aug_data = aug_data

        augmented_data = data.copy()
        augmented_data[start_pos:end_pos] = aug_data

        return augmented_data
