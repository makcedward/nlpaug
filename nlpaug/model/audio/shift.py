# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation
import numpy as np

from nlpaug.model.audio import Audio


class Shift(Audio):
    def validate(self, direction):
        if direction not in ['left', 'right', 'random']:
            raise ValueError(
                'shift_direction should be either left, right or both while {} is passed.'.format(direction))

    def manipulate(self, data, shift):
        aug_data = np.roll(data.copy(), shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            aug_data[:shift] = 0
        else:
            aug_data[shift:] = 0
        return aug_data
