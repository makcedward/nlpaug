"""
    Augmenter that apply pitch adjustment operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class PitchAug(AudioAugmenter):
    """
    :param int sampling_rate: Sampling rate of input audio.
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param int duration: Duration of augmentation (in second). Default value is None. If value is provided. `coverage`
        value will be ignored.
    :param tuple factor: Input data pitch will be increased (decreased). Augmented value will be picked
            within the range of this tuple value. Pitch will be reduced if value is between 0 and 1.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.PitchAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate, zone=(0.2, 0.8), coverage=1., duration=None, 
        factor=(-10, 10), name='Pitch_Aug', verbose=0, stateless=True):
        super().__init__(action=Action.SUBSTITUTE, zone=zone, coverage=coverage, factor=factor, 
            duration=duration, name=name, device='cpu', verbose=verbose, stateless=stateless)

        self.sampling_rate = sampling_rate
        self.model = nma.Pitch()

    def substitute(self, data):
        pitch_level = self.get_random_factor()
        start_pos, end_pos = self.get_augment_range_by_coverage(data)

        if not self.stateless:
            self.start_pos, self.end_pos, self.aug_factor = start_pos, end_pos, pitch_level

        return self.model.manipulate(data, start_pos, end_pos, pitch_level, self.sampling_rate)

