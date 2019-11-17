# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation
import numpy as np

from nlpaug.model.audio import Audio


class Shift(Audio):
    def __init__(self, sampling_rate, duration=3,
                 direction='random', stateless=True):
        """
        :param int sampling_rate: Sampling rate of input audio.
        :param float duration: Max shifting segment (in second)
        :param str direction: Shifting segment to left, right or one of them. Value can be 'left', 'right' or 'random'
        """

        super().__init__(duration=duration, sampling_rate=sampling_rate, stateless=stateless)
        # TODO: remove `both` after 0.0.12
        if direction in ['left', 'right', 'random', 'both']:
            self.direction = direction
        else:
            raise ValueError(
                'shift_direction should be either left, right or both while {} is passed.'.format(direction))

    def manipulate(self, data):
        aug_shift = int(self.sampling_rate * self.duration)
        if self.direction == 'right':
            aug_shift = -aug_shift
        elif self.direction == 'random':
            direction = np.random.randint(0, 2)
            if direction == 1:
                aug_shift = -aug_shift

        augmented_data = np.roll(data, aug_shift)

        # Set to silence for heading/ tailing
        if aug_shift > 0:
            augmented_data[:aug_shift] = 0
        else:
            augmented_data[aug_shift:] = 0
        return augmented_data
