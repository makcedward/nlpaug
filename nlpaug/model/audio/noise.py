import random
import numpy as np
import math

from nlpaug.model.audio import Audio


class Noise(Audio):
    COLOR_NOISES = ['white', 'pink', 'red', 'brown', 'brownian', 'blue', 'azure', 'violet', 'purple']

    def validate(self, color):
        if color not in self.COLOR_NOISES + ['random']:
            raise ValueError('Only support {} while `{}` is passed'.format(self.COLOR_NOISES+['random'], self.color))

    def get_noise_and_color(self, aug_segment_size, noises, color):
        if noises is None:
            noise, _color = self.color_noise(aug_segment_size, color)
        else:
            noise, _color = self.background_noise(aug_segment_size, noises), color

        return noise, _color

    def color_noise(self, segment_size, color):
        # https://en.wikipedia.org/wiki/Colors_of_noise
        uneven = segment_size % 2
        fft_size = segment_size // 2 + 1 + uneven
        noise_fft = np.random.randn(fft_size)
        color_noise = np.linspace(1, fft_size, fft_size)

        if color == 'random':
            _color = np.random.choice(self.COLOR_NOISES)
        else:
            _color = color

        if _color == 'white':
            pass  # no color noise
        else:
            if _color == 'pink':
                color_noise = color_noise ** (-1)  # 1/f
            elif _color in ['red', 'brown', 'brownian']:
                color_noise = color_noise ** (-2)  # 1/f^2
            elif _color in ['blue', 'azure']:
                pass  # f
            elif _color in ['violet', 'purple']:
                color_noise = color_noise ** 2  # f^2

            noise_fft = noise_fft * color_noise

        if uneven:
            noise_fft = noise_fft[:-1]

        noise = np.fft.irfft(noise_fft)
        return noise, color_noise

    def background_noise(self, segment_size, noises):
        # https://arxiv.org/pdf/1608.04363.pdf
        noise = random.sample(noises, 1)[0]

        # Get noise segment
        if len(noise) >= segment_size:
            noise_segment = noise[:segment_size]
        else:
            noise_segment = noise.copy()
            for _ in range(math.ceil(segment_size/len(noise))-1):
                noise_segment = np.append(noise_segment, noise)
            noise_segment = noise_segment[:segment_size]

        return noise_segment

    def pad(self, data, noise):
        if len(data) - len(noise) == 0:
            start_pos = 0
        else:
            start_pos = np.random.randint(0, len(data) - len(noise))

        prefix_padding = np.array([0] * start_pos)
        suffix_padding = np.array([0] * (len(data) - len(noise) - start_pos))
        return np.append(np.append(prefix_padding, noise), suffix_padding)

    def manipulate(self, data, start_pos, end_pos, noise):
        noise = self.pad(data[start_pos:end_pos], noise)

        return np.concatenate((data[:start_pos], (data[start_pos:end_pos]+noise), data[end_pos:]), axis=0).astype(type(data[0]))
