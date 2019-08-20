import numpy as np

from nlpaug.model.audio import Audio


class Noise(Audio):
    """
        Reference: https://www.kaggle.com/CVxTz/audio-data-augmentation
    """

    def __init__(self, noise_factor):
        """

        :param noise_factor: Strength of noise
        """
        super(Noise, self).__init__()

        self.noise_factor = noise_factor

    def manipulate(self, data):
        noise = np.random.randn(len(data))
        augmented_data = data + self.noise_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data
