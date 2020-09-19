# Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation

try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass
import numpy as np

from nlpaug.model.audio import Audio


class Pitch(Audio):
    def __init__(self):
        super().__init__()
        try:
            import librosa
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed librosa library. Install it via `pip install librosa`')

    def manipulate(self, data, start_pos, end_pos, pitch_level, sampling_rate):
        aug_data = data.copy()
        aug_data[start_pos:end_pos] = librosa.effects.pitch_shift(
            aug_data[start_pos:end_pos], sampling_rate, pitch_level)

        return aug_data
