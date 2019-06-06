from nlpaug.util import Action
from nlpaug.flow import Pipeline


class Sometimes(Pipeline):
    # TODO: Using epoch to implement 1-to-many
    def __init__(self, flow=None, name='Sometimes_Pipeline', pipeline_p=0.2, aug_p=1, verbose=0):
        Pipeline.__init__(self, name=name, action=Action.SOMETIMES,
                          flow=flow, epoch=1, aug_min=-1, aug_p=aug_p, verbose=verbose)

        self.pipeline_p = pipeline_p

    def augment(self, text):
        results = []

        for _ in range(self.epoch):
            augmented_text = text[:]
            for aug in self:
                if self.pipeline_p < self.prob():
                    continue

                augmented_text = aug.augment(augmented_text)

            results.append(augmented_text)

        return results[0]
