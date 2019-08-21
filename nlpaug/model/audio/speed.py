import librosa
import numpy as np

from nlpaug.model.audio import Audio

"""
    Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation
    A wrapper of librosa.effects.time_stretch
"""

# TODO: Validation


class Speed(Audio):
    """
    Adjusting speed

    :param speed_range: Factor for time stretch. Audio will be slowing down if value is between 0 and 1.
        Audio will be speed up if value is larger than 1.
    """
    def __init__(self, speed_range):
        super(Speed, self).__init__()

        # if speed_factor < 0:
        #     raise ValueError(
        #         'speed_factor should be positive number while {} is passed.'.format(speed_factor))
        self.speed_range = speed_range

    def manipulate(self, data):
        speeds = [round(i, 1) for i in np.arange(self.speed_range[0], self.speed_range[1], 0.1)]
        speed = speeds[np.random.randint(len(speeds))]

        return librosa.effects.time_stretch(data, speed)
