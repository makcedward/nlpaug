# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation

try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass
import numpy as np

from nlpaug.model.audio import Audio


class Speed(Audio):
    def __init__(self):
        super().__init__()
        try:
            import librosa
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed librosa library. Install it via `pip install librosa`')

    def manipulate(self, data, start_pos, end_pos, speed):
        aug_data = librosa.effects.time_stretch(data[start_pos:end_pos], speed)
        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0)
