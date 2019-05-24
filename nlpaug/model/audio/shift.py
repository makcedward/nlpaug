import numpy as np

from nlpaug.model.audio import Audio


class Shift(Audio):
    def __init__(self, sampling_rate, shift_max=2, shift_direction='both'):
        super(Shift, self).__init__()

        self.sampling_rate = sampling_rate
        self.shift_max = shift_max

        if shift_direction in ['left', 'right', 'both']:
            self.shift_direction = shift_direction
        else:
            raise ValueError(
                'shift_direction should be either left, right or both while {} is passed.'.format(shift_direction))

    def manipulate(self, data):
        shift = np.random.randint(self.sampling_rate * self.shift_max)
        if self.shift_direction == 'right':
            shift = -shift
        elif self.shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift

        augmented_data = np.roll(data, shift)

        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data
