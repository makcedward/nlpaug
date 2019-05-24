import numpy as np

from nlpaug.model.audio import Audio


class Noise(Audio):
    def __init__(self, nosie_factor):
        super(Noise, self).__init__()

        self.nosie_factor = nosie_factor

    def manipulate(self, data):
        noise = np.random.randn(len(data))
        augmented_data = data + self.nosie_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data
