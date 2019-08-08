from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class CropAug(AudioAugmenter):
    def __init__(self, sampling_rate, crop_range=(0.2, 0.8), crop_factor=2, name='Crop_Aug', verbose=0):
        """

        :param sampling_rate: sampling rate of input audio
        :param crop_range: Range of applying crop operation. Default value is (0.2, 0.8)
            It means that first 20% and last 20% of data will not be excluded from augment operation. Augment operation
            will be applied to clip of rest of 60% time.
        :param crop_factor: duration of cropping period (in second)
        :param name: Name of this augmenter
        """

        super().__init__(
            action=Action.DELETE, name=name, verbose=verbose)
        self.model = self.get_model(sampling_rate, crop_range, crop_factor)

    def delete(self, data):
        return self.model.manipulate(data)

    def get_model(self, sampling_rate, crop_range, crop_factor):
        return nma.Corp(sampling_rate, crop_range, crop_factor)
