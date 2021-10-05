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

    def __init__(self, sampling_rate, duration=3, direction='random', shift_direction='random', 
        name='Shift_Aug', verbose=0, stateless=True):
        super().__init__(action=Action.SUBSTITUTE, name=name, duration=duration, device='cpu', verbose=verbose, 
            stateless=stateless)

        self.sampling_rate = sampling_rate
        self.direction = direction
        self.shift_direction = shift_direction
        self.model = nma.Shift()

        self.model.validate(shift_direction)

    def _get_aug_shift(self):
        aug_shift = int(self.sampling_rate * self.duration)
        if self.direction == 'right':
            return -aug_shift
        elif self.direction == 'random':
            direction = self.sample(4)-1
            if direction == 1:
                return -aug_shift

        return aug_shift

    def substitute(self, data):
        aug_shift = self._get_aug_shift()

        if not self.stateless:
            self.aug_factor = aug_shift

        return self.model.manipulate(data, aug_shift)
