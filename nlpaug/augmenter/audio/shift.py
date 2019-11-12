"""
    Augmenter that apply shifting operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class ShiftAug(AudioAugmenter):
    """
    :param int sampling_rate: Sampling rate of input audio.
    :param float duration: Max shifting segment (in second)
    :param str direction: Shifting segment to left, right or one of them. Value can be 'left', 'right' or 'random'
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.ShiftAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate, duration=3, direction='random',
                 shift_max=3, shift_direction='both',
                 name='Shift_Aug', verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)

        if shift_direction != 'both':
            print(WarningMessage.DEPRECATED.format('shift_direction', '0.0.12', 'direction'))
            direction = shift_direction
        if shift_max != 3:
            print(WarningMessage.DEPRECATED.format('shift_max', '0.0.12', 'duration'))
            duration = shift_max

        self.model = self.get_model(sampling_rate, duration, direction)

    @classmethod
    def get_model(cls, sampling_rate, duration, direction):
        return nma.Shift(sampling_rate, duration, direction)
