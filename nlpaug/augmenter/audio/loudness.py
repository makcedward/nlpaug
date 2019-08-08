from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class LoudnessAug(AudioAugmenter):
    def __init__(self, loudness_factor=(0.5, 2), name='Loudness_Aug', verbose=0):
        """

        :param loudness_factor: Input data volume will be increased (decreased). Augmented value will be picked
            within the range of this tuple value. If volume will be reduced if value is between 0 and 1.
        :param name: Name of this augmenter
        """

        super().__init__(
            action=Action.SUBSTITUTE, name=name, verbose=verbose)
        self.model = self.get_model(loudness_factor)

    def get_model(self, loudness_factor):
        return nma.Loudness(loudness_factor)
