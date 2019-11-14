import numpy as np
import librosa
from nlpaug.model.audio import Audio


class Vtlp(Audio):
    # https://pdfs.semanticscholar.org/3de0/616eb3cd4554fdf9fd65c9c82f2605a17413.pdf
    def __init__(self, sampling_rate, zone=(0.2, 0.8), coverage=0.1, duration=None, factor=(0.9, 1.1), fhi=4800,
                 stateless=True):
        """
        :param int sampling_rate: Sampling rate of input audio. Mandatory if duration is provided.
        :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
            augmentation
        will be applied in first 20% and last 20% of whole audio.
        :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
            operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
            zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
            augmented.
        :param int duration: Duration of augmentation (in second). Default value is None. If value is provided.
            `coverage` value will be ignored.
        :param int fhi: Boundary frequency. Default value is 4800.
        :param tuple factor: Warping factor
        """
        super().__init__(zone=zone, coverage=coverage, duration=duration, sampling_rate=sampling_rate,
                         stateless=stateless, factor=factor)
        self.fhi = fhi

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

    def get_warping_level(self):
        return np.random.uniform(self.factor[0], self.factor[1])

    def manipulate(self, data):
        if self.duration is None:
            start_pos, end_pos = self.get_augment_range_by_coverage(data)
        else:
            start_pos, end_pos = self.get_augment_range_by_duration(data)

        factor = self.get_warping_level()
        aug_data = self._manipulate(data[start_pos:end_pos], sampling_rate=self.sampling_rate, factor=factor)

        if not self.stateless:
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.aug_factor = factor
            self.aug_data = aug_data

        return np.concatenate((data[:start_pos], aug_data, data[end_pos:]), axis=0).astype(type(data[0]))

        # if start_pos > 0:
        #     aug_data = np.concatenate((data[:start_pos], aug_data), axis=0)
        # if end_pos < len(data):
        #     aug_data = np.concatenate((aug_data, data[end_pos:]), axis=0)
        #
        # return aug_data.astype(type(data[0]))
