"""
    Augmenter that apply shifting operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class ShiftAug(AudioAugmenter):
    """
    Augmenter that crop segment of audio by random values between crop_range variable.

    :param int sampling_rate: sampling rate of input audio
    :param int shift_max: Max shifting segment (in second)
    :param str shift_direction: Shifting segment to left, right or one of them. Value can be 'left', 'right' or 'both'.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.ShiftAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate, shift_max=3, shift_direction='both', name='Shift_Aug', verbose=0):
        super(ShiftAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)
        self.model = self.get_model(sampling_rate, shift_max, shift_direction)

    @classmethod
    def get_model(cls, sampling_rate, shift_max, shift_direction):
        return nma.Shift(sampling_rate, shift_max, shift_direction)