from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class NoiseAug(AudioAugmenter):
    def __init__(self, nosie_factor=0.01, name='Noise_Aug'):
        super(NoiseAug, self).__init__(
            action=Action.SUBSTITUTE, name=name)
        self.model = self.get_model(nosie_factor)

    def get_model(self, nosie_factor):
        return nma.Noise(nosie_factor)
