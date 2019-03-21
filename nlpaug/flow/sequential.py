from nlpaug.util import Action
from nlpaug.flow import Pipeline


class Sequential(Pipeline):
    # TODO: Using epcoh to implement 1-to-many
    def __init__(self, flow=None, name='Sequential_Pipeline', aug_p=1):
        Pipeline.__init__(self, name=name, action=Action.SEQUENTIAL,
                          flow=flow, epoch=1, aug_min=-1, aug_p=aug_p)

    def augment(self, tokens):
        results = []
        for _ in range(self.epoch):
            augmented_inputs = tokens.copy()
            for aug in self:
                augmented_inputs = aug.augment(augmented_inputs)
            results.append(augmented_inputs)

        return results[0]
