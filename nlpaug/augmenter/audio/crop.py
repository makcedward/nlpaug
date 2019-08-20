"""
    Augmenter that apply cropping operation to audio.
"""

from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class CropAug(AudioAugmenter):
    """
    Augmenter that crop segment of audio by random values between crop_range variable.

    :param int sampling_rate: sampling rate of input audio
    :param tuple crop_range: Range of applying crop operation. Default value is (0.2, 0.8)
        It means that first 20% and last 20% of data will be excluded from augment operation selection.
    :param int crop_factor: duration of cropping period (in second)
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.audio as naa
    >>> aug = naa.CropAug(sampling_rate=44010)
    """

    def __init__(self, sampling_rate, crop_range=(0.2, 0.8), crop_factor=2, name='Crop_Aug', verbose=0):
        super().__init__(
            action=Action.DELETE, name=name, verbose=verbose)
        self.model = self.get_model(sampling_rate, crop_range=crop_range, crop_factor=crop_factor)

    def delete(self, data):
        return self.model.manipulate(data)

    @classmethod
    def get_model(cls, sampling_rate, crop_range, crop_factor):
        return nma.Corp(sampling_rate, crop_range, crop_factor)
