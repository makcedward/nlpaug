"""
    Augmenter that apply speed adjustment operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class SpeedAug(AudioAugmenter):
    """
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param int factor: Range of applying speed adjustment operation. Default value is (0.5, 2)
        Factor for time stretch. Audio will be slowing down if value is between 0 and 1.
    :param tuple speed_range: Deprecated. Use `factor` indeed
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.ShiftAug()
    """

    def __init__(self, zone=(0.2, 0.8), coverage=1., duration=None,
                 factor=(0.5, 2),
                 speed_range=(0.5, 2), name='Speed_Aug', verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)

        if speed_range != (0.5, 2):
            print(WarningMessage.DEPRECATED.format('speed_range', '0.0.12', 'factor'))
            factor = speed_range

        self.model = self.get_model(zone, coverage, duration, factor)

    @classmethod
    def get_model(cls, zone, coverage, duration, factor):
        return nma.Speed(zone, coverage, duration, factor)
