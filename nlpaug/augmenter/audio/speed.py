from nlpaug.augmenter.audio import AudioAugmenter
from nlpaug.util import Action
import nlpaug.model.audio as nma


class SpeedAug(AudioAugmenter):
    def __init__(self, speed_factor, name='Speed_Aug'):
        """
        :param speed_factor: integer
            Factor for time stretch. Audio will be slowing down if value is between 0 and 1.
            Audio will be speed up if value is larger than 1.
        :param name: Name of this augmenter
        """
        super(SpeedAug, self).__init__(
            action=Action.SUBSTITUTE, name=name)
        self.model = self.get_model(speed_factor)

    def get_model(self, speed_factor):
        return nma.Speed(speed_factor)