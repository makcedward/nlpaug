# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation

try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass
import numpy as np

from nlpaug.model.audio import Audio


class Pitch(Audio):
    """
    Adjusting pitch

    :param sampling_rate: Sampling rate of input audio
    :param pitch_range: Number of half-steps that shifting audio
    """
    def __init__(self, sampling_rate, pitch_range):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.pitch_range = pitch_range

        try:
            librosa
        except NameError:
            raise ImportError('Missed librosa library. Install it via `pip install librosa`')

    def manipulate(self, data):
        n_step = np.random.randint(self.pitch_range[0], self.pitch_range[1])

        return librosa.effects.pitch_shift(data, self.sampling_rate, n_step)
