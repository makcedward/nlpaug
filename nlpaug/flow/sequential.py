from nlpaug.util import Action
from nlpaug.flow import Pipeline


class Sequential(Pipeline):
    # TODO: Using epoch to implement 1-to-many
    def __init__(self, flow=None, name='Sequential_Pipeline', aug_p=1):
        Pipeline.__init__(self, name=name, action=Action.SEQUENTIAL,
                          flow=flow, epoch=1, aug_min=-1, aug_p=aug_p)

    def augment(self, text):
        results = []
        for _ in range(self.epoch):
            augmented_text = text[:]
            for aug in self:
                augmented_text = aug.augment(augmented_text)

            results.append(augmented_text)

        return results[0]
