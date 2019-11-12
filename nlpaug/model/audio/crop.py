import numpy as np

from nlpaug.model.audio import Audio


class Crop(Audio):
    def __init__(self, sampling_rate=None, zone=(0.2, 0.8), coverage=0.1, duration=None, stateless=True):
        """
        :param int sampling_rate: Sampling rate of input audio. Mandatory if duration is provided.
        :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
            augmentation
        will be applied in first 20% and last 20% of whole audio.
        :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
            operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
            zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
            augmented.
        :param int duration: Duration of augmentation (in second). Default value is None. If value is provided.
            `coverage` value will be ignored.
        """
        super().__init__(zone=zone, coverage=coverage, duration=duration, sampling_rate=sampling_rate,
                         stateless=stateless)

    def manipulate(self, data):
        if self.duration is None:
            start_pos, end_pos = self.get_augment_range_by_coverage(data)
        else:
            start_pos, end_pos = self.get_augment_range_by_duration(data)

        if not self.stateless:
            self.start_pos = start_pos
            self.end_pos = end_pos

        augmented_data = np.delete(data, np.s_[start_pos:end_pos])
        return augmented_data
