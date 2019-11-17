# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation

try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass
import numpy as np

from nlpaug.model.audio import Audio


class Pitch(Audio):
    def __init__(self, sampling_rate, zone=(0.2, 0.8), coverage=1., duration=None,
                 factor=(-10, 10), stateless=False):
        """
        :param int sampling_rate: Sampling rate of input audio. Mandatory if duration is provided.
        :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
            augmentation
        will be applied in first 20% and last 20% of whole audio.
        :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
            operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
            zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
            augmented.
        :param int duration: Duration of augmentation (in second). Default value is None. If value is provided. `coverage`
            value will be ignored.
        :param tuple pitch_range: Deprecated. Use `factor` indeed
        :param str name: Name of this augmenter
        """
        super().__init__(zone=zone, coverage=coverage, duration=duration, sampling_rate=sampling_rate,
                         factor=factor, stateless=stateless)
        try:
            librosa
        except NameError:
            raise ImportError('Missed librosa library. Install it via `pip install librosa`')

    def get_pitch_level(self):
        return np.random.randint(self.factor[0], self.factor[1])

    def manipulate(self, data):
        n_step = self.get_pitch_level()
        start_pos, end_pos = self.get_augment_range_by_coverage(data)

        aug_data = librosa.effects.pitch_shift(data[start_pos:end_pos], self.sampling_rate, n_step)

        if not self.stateless:
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.aug_data = aug_data

        augmented_data = data.copy()
        augmented_data[start_pos:end_pos] = aug_data

        return augmented_data
