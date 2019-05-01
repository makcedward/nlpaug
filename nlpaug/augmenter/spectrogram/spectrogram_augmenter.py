from nlpaug.util import Method
from nlpaug import Augmenter


class SpectrogramAugmenter(Augmenter):
    def __init__(self, action, name='Spectrogram_Aug', aug_min=1, aug_p=0.3):
        super(SpectrogramAugmenter, self).__init__(
            name=name, method=Method.SPECTROGRAM, action=action, aug_min=aug_min)
        self.aug_p = aug_p

