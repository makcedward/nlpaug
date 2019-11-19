import random
import numpy as np
import math

from nlpaug.model.audio import Audio


class Noise(Audio):
    COLOR_NOISES = ['white', 'pink', 'red', 'brown', 'brownian', 'blue', 'azure', 'violet', 'purple']

    def __init__(self, zone=(0.2, 0.8), coverage=1.,
                 color='white', noises=None, stateless=True):
        """
        :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
            augmentation
        will be applied in first 20% and last 20% of whole audio.
        :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
            operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
            zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
            augmented.
        :param str color: Colors of noise. Supported 'white', 'pink', 'red', 'brown', 'brownian', 'blue', 'azure',
            'violet', 'purple' and 'random'. If 'random' is used, noise color will be picked randomly in each augment.
        :param list noises: Background noises for noise injection. You can provide more than one background noise and
            noise will be picked randomly. Expected format is list of numpy array. If this value is provided. `color`
            value will be ignored
        """
        super().__init__(zone=zone, coverage=coverage, stateless=stateless)

        self.color = color
        self.noises = noises

    def validate(self):
        if self.color not in self.COLOR_NOISES + ['random']:
            raise ValueError('Only support {} while `{}` is passed'.format(self.COLOR_NOISES+['random'], self.color))

    def color_noise(self, segment_size):
        # https://en.wikipedia.org/wiki/Colors_of_noise
        uneven = segment_size % 2
        fft_size = segment_size // 2 + 1 + uneven
        noise_fft = np.random.randn(fft_size)
        color_noise = np.linspace(1, fft_size, fft_size)

        if self.color == 'random':
            color = np.random.choice(self.COLOR_NOISES)
        else:
            color = self.color
        if color == 'white':
            pass  # no color noise
        else:
            if color == 'pink':
                color_noise = color_noise ** (-1)  # 1/f
            elif color in ['red', 'brown', 'brownian']:
                color_noise = color_noise ** (-2)  # 1/f^2
            elif color in ['blue', 'azure']:
                pass  # f
            elif color in ['violet', 'purple']:
                color_noise = color_noise ** 2  # f^2

            noise_fft = noise_fft * color_noise

        if uneven:
            noise_fft = noise_fft[:-1]

        noise = np.fft.irfft(noise_fft)
        return noise, color_noise

    def background_noise(self, segment_size):
        # https://arxiv.org/pdf/1608.04363.pdf
        noise = random.sample(self.noises, 1)[0]

        # Get noise segment
        if len(noise) >= segment_size:
            noise_segment = noise[:segment_size]
        else:
            noise_segment = noise.copy()
            for _ in range(math.ceil(segment_size/len(noise))-1):
                noise_segment = np.append(noise_segment, noise)
            noise_segment = noise_segment[:segment_size]

        return noise_segment

    def manipulate(self, data):
        aug_segment_size = self.get_augmentation_segment_size(data)
        if self.noises is None:
            noise, color = self.color_noise(aug_segment_size)

            if not self.stateless:
                self.aug_factor = color
        else:
            noise = self.background_noise(aug_segment_size)

        if not self.stateless:
            self.aug_data = noise

        noise = self.pad(data, noise)

        return (data + noise).astype(type(data[0]))
