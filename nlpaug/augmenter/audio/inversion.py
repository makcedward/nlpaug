"""
    Augmenter that apply polarity inversion to audio. 
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class PolarityInverseAug(AudioAugmenter):
    """
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.PolarityInverseAug()
    """

    def __init__(self, name='PolarityInverse_Aug', verbose=0, stateless=True):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose, 
            stateless=stateless)

        self.model = nma.PolarityInversion()

    def substitute(self, data):
        return self.model.manipulate(data)
