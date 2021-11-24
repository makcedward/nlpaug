"""
    Augmenter that apply speed adjustment operation to audio.
"""

import numpy as np

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
    :param tuple factor: Input data speed will be increased (decreased). Augmented value will be picked
        within the range of this tuple value. Speed will be reduced if value is between 0 and 1.
    :param tuple speed_range: Deprecated. Use `factor` indeed
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.ShiftAug()
    """

    def __init__(self, zone=(0.2, 0.8), coverage=1., factor=(0.5, 2), name='Speed_Aug', verbose=0, 
        stateless=True):
        super().__init__(action=Action.SUBSTITUTE, name=name, zone=zone, coverage=coverage, 
            factor=factor, device='cpu', verbose=verbose, stateless=stateless)

        self.model = nma.Speed()

    def substitute(self, data):
        speed_level = self.get_random_factor()
        start_pos, end_pos = self.get_augment_range_by_coverage(data)

        if not self.stateless:
            self.start_pos, self.end_pos, self.aug_factor = start_pos, end_pos, speed_level

        return self.model.manipulate(data, start_pos=start_pos, end_pos=end_pos, speed=speed_level)
        
