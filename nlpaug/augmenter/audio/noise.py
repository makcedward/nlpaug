from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class NoiseAug(AudioAugmenter):
    def __init__(self, noise_factor=0.01, name='Noise_Aug', verbose=0):
        """

        :param noise_factor: Strength of noise
        :param name: Name of this augmenter
        """
        super(NoiseAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, verbose=verbose)
        self.model = self.get_model(noise_factor)

    def get_model(self, noise_factor):
        return nma.Noise(noise_factor)
