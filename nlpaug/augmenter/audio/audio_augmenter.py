from nlpaug.util import Method
from nlpaug import Augmenter


class AudioAugmenter(Augmenter):
    def __init__(self, action, name='Audio_Aug', verbose=0):
        super(AudioAugmenter, self).__init__(
            name=name, method=Method.AUDIO, action=action, aug_min=1, verbose=verbose)

    def substitute(self, data):
        return self.model.manipulate(data)
