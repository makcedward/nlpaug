import librosa

from nlpaug.model.audio import Audio

"""
    A wrapper of librosa.effects.time_stretch
"""


class Speed(Audio):
    def __init__(self, speed_factor):
        super(Speed, self).__init__()

        if speed_factor < 0:
            raise ValueError(
                'speed_factor should be positive number while {} is passed.'.format(speed_factor))
        self.speed_factor = speed_factor

    def manipulate(self, data):
        return librosa.effects.time_stretch(data, self.speed_factor)
