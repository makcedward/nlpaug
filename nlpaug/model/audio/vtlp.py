import numpy as np
try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.audio import Audio


class Vtlp(Audio):
    # https://pdfs.semanticscholar.org/3de0/616eb3cd4554fdf9fd65c9c82f2605a17413.pdf
    def __init__(self):
        super().__init__()

        try:
            import librosa
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed librosa library. Install import librosa by `pip install librosa`')

    @classmethod
    def get_scale_factors(cls, freq_dim, sampling_rate, fhi=4800, alpha=0.9):
        factors = []
        freqs = np.linspace(0, 1, freq_dim)

        scale = fhi * min(alpha, 1)
        f_boundary = scale / alpha
        half_sr = sampling_rate / 2

        for f in freqs:
            f *= sampling_rate
            if f <= f_boundary:
                factors.append(f * alpha)
            else:
                warp_freq = half_sr - (half_sr - scale) / (half_sr - scale / alpha) * (half_sr - f)
                factors.append(warp_freq)

        return np.array(factors)

    # https://github.com/YerevaNN/Spoken-language-identification/blob/master/augment_data.py#L26
    def _manipulate(self, audio, sampling_rate, factor):
        stft = librosa.core.stft(audio)
        time_dim, freq_dim = stft.shape
        data_type = type(stft[0][0])

        factors = self.get_scale_factors(freq_dim, sampling_rate, alpha=factor)
        factors *= (freq_dim - 1) / max(factors)
        new_stft = np.zeros([time_dim, freq_dim], dtype=data_type)

        for i in range(freq_dim):
            # first and last freq
            if i == 0 or i + 1 >= freq_dim:
                new_stft[:, i] += stft[:, i]
            else:
                warp_up = factors[i] - np.floor(factors[i])
                warp_down = 1 - warp_up
                pos = int(np.floor(factors[i]))

                new_stft[:, pos] += warp_down * stft[:, i]
                new_stft[:, pos+1] += warp_up * stft[:, i]

        return librosa.core.istft(new_stft)

    def manipulate(self, data, start_pos, end_pos, sampling_rate, warp_factor):
        aug_data = self._manipulate(data[start_pos:end_pos], sampling_rate=sampling_rate, factor=warp_factor)

        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0).astype(type(data[0]))