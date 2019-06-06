from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class PitchAug(AudioAugmenter):
    def __init__(self, sampling_rate, pitch_factor, name='Pitch_Aug', verbose=0):
        """
        :param pitch_factor: integer
        :param name: Name of this augmenter
        """
        super(PitchAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, verbose=verbose)
        self.model = self.get_model(sampling_rate, pitch_factor)

    def get_model(self, sampling_rate, pitch_factor):
        return nma.Pitch(sampling_rate, pitch_factor)