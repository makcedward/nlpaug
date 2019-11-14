"""
    Augmenter that apply pitch adjustment operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class PitchAug(AudioAugmenter):
    """
    :param int sampling_rate: Sampling rate of input audio. Mandatory if duration is provided.
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param int duration: Duration of augmentation (in second). Default value is None. If value is provided. `coverage`
        value will be ignored.
    :param tuple pitch_range: Deprecated. Use `factor` indeed
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.PitchAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate, zone=(0.2, 0.8), coverage=1., duration=None,
                 factor=(-10, 10), pitch_range=(-10, 10), name='Pitch_Aug', verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)

        if pitch_range != (-10, 10):
            print(WarningMessage.DEPRECATED.format('pitch_range', '0.0.12', 'factor'))
            factor = pitch_range

        self.model = self.get_model(sampling_rate, zone, coverage, duration, factor)

    @classmethod
    def get_model(cls, sampling_rate, zone, coverage, duration, factor):
        return nma.Pitch(sampling_rate, zone, coverage, duration, factor)
