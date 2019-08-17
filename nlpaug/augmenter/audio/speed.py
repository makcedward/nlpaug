"""
    Augmenter that apply speed adjustment operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class SpeedAug(AudioAugmenter):
    """
    Augmenter that crop segment of audio by random values between crop_range variable.

    :param tuple speed_range: Range of applying speed adjustment operation. Default value is (0.5, 2)
        Factor for time stretch. Audio will be slowing down if value is between 0 and 1.
        Audio will be speed up if value is larger than 1.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.ShiftAug()
    """

    def __init__(self, speed_range=(0.5, 2), name='Speed_Aug', verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, verbose=verbose)
        self.model = self.get_model(speed_range)

    @classmethod
    def get_model(cls, speed_range):
        return nma.Speed(speed_range)
