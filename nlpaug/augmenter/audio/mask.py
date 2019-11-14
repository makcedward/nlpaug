"""
    Augmenter that apply mask operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
import nlpaug.model.audio as nma
from nlpaug.util import Action, WarningMessage


class MaskAug(AudioAugmenter):
    """
    :param int sampling_rate: Sampling rate of input audio. Mandatory if duration is provided.
    :param tuple zone: Assign a zone for augmentation. Default value is (0.2, 0.8) which means that no any
        augmentation will be applied in first 20% and last 20% of whole audio.
    :param float coverage: Portion of augmentation. Value should be between 0 and 1. If `1` is assigned, augment
        operation will be applied to target audio segment. For example, the audio duration is 60 seconds while
        zone and coverage are (0.2, 0.8) and 0.7 respectively. 42 seconds ((0.8-0.2)*0.7*60) audio will be
        augmented.
    :param float duration: Duration of augmentation (in second). Default value is None. If value is provided. `coverage`
        value will be ignored.
    :param bool mask_with_noise: If it is True, targeting area will be replaced by noise. Otherwise, it will be
            replaced by 0.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.MaskAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate=None, zone=(0.2, 0.8), coverage=1.,
                 duration=(0.2, 0.8),
                 mask_range=(0.2, 0.8), mask_factor=2, mask_with_noise=True,
                 name='Mask_Aug', verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)

        if mask_range != (0.2, 0.8):
            print(WarningMessage.DEPRECATED.format('mask_range', '0.0.12', 'zone'))
            zone = mask_range
        if mask_factor != 2:
            print(WarningMessage.DEPRECATED.format('mask_factor', '0.0.12', 'duration'))
            duration = mask_factor

        self.model = self.get_model(sampling_rate, zone, coverage, duration, mask_with_noise)

    @classmethod
    def get_model(cls, sampling_rate, zone, coverage, duration, mask_with_noise):
        return nma.Mask(sampling_rate, zone, coverage, duration, mask_with_noise)
