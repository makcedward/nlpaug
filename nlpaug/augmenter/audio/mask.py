"""
    Augmenter that apply mask operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class MaskAug(AudioAugmenter):
    """
    Augmenter that crop segment of audio by random values between crop_range variable.

    :param int sampling_rate: sampling rate of input audio
    :param tuple mask_range: Range of applying mask operation. Default value is (0.2, 0.8)
        It means that first 20% and last 20% of data will not be excluded from augment operation. Augment operation
        will be applied to clip of rest of 60% time.
    :param int mask_factor: duration of masking period (in second)
    :param bool mask_with_noise: If it is True, targeting area will be replaced by noise. Otherwise, it will be
            replaced by 0.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.MaskAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate, mask_range=(0.2, 0.8), mask_factor=2, mask_with_noise=True,
                 name='Mask_Aug', verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, name=name, device='cpu', verbose=verbose)
        self.model = self.get_model(sampling_rate, mask_range, mask_factor, mask_with_noise)

    @classmethod
    def get_model(cls, sampling_rate, mask_range, mask_factor, mask_with_noise):
        return nma.Mask(sampling_rate, mask_range, mask_factor, mask_with_noise)
