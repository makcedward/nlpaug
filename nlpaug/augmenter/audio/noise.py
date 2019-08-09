"""
    Augmenter that apply noise injection operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


# TODO: Apply noise injection to segment of audio only

class NoiseAug(AudioAugmenter):
    """
    Augmenter that crop segment of audio by random values between crop_range variable.

    :param int noise_factor: Strength of noise
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.NoiseAug()
    """

    def __init__(self, noise_factor=0.01, name='Noise_Aug', verbose=0):
        super(NoiseAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, verbose=verbose)
        self.model = self.get_model(noise_factor)

    def get_model(self, noise_factor):
        return nma.Noise(noise_factor)
