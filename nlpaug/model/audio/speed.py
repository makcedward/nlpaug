# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation

try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass
import numpy as np

from nlpaug.model.audio import Audio


class Speed(Audio):
    """
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param int factor: Range of applying speed adjustment operation. Default value is (0.5, 2)
        Factor for time stretch. Audio will be slowing down if value is between 0 and 1.
    """
    def __init__(self, zone=(0.2, 0.8), coverage=1., duration=None,
                 factor=(-10, 10), stateless=False):
        super().__init__(zone=zone, coverage=coverage, duration=duration,
                         factor=factor, stateless=stateless)
        try:
            librosa
        except NameError:
            raise ImportError('Missed librosa library. Install it via `pip install librosa`')

    def get_speed_level(self):
        speeds = [round(i, 1) for i in np.arange(self.factor[0], self.factor[1], 0.1)]
        speeds = [s for s in speeds if s != 1.0]
        return speeds[np.random.randint(len(speeds))]

    def manipulate(self, data):
        speed = self.get_speed_level()
        start_pos, end_pos = self.get_augment_range_by_coverage(data)

        aug_data = librosa.effects.time_stretch(data[start_pos:end_pos], speed)

        if not self.stateless:
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.aug_data = aug_data
            self.aug_factor = speed

        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0)
