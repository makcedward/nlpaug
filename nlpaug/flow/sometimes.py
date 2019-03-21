from nlpaug.util import Action
from nlpaug.flow import Pipeline


class Sometimes(Pipeline):
    # TODO: Using epcoh to implement 1-to-many
    def __init__(self, flow=None, name='Sometimes_Pipeline', pipeline_p=0.2, aug_p=1):
        Pipeline.__init__(self, name=name, action=Action.SOMETIMES,
                          flow=flow, epoch=1, aug_min=-1, aug_p=aug_p)

        self.pipeline_p = pipeline_p

    def augment(self, tokens):
        results = []

        for _ in range(self.epoch):
            augmented_inputs = tokens.copy()

            for aug in self:
                if self.pipeline_p < self.prob():
                    continue

                aug_idxes = self.generate_aug_idxes(tokens)

                for j, w in enumerate(augmented_inputs):
                    if j in aug_idxes:
                        augmented_inputs[j] = aug.augment([w])[0]
            results.append(augmented_inputs)

        return results[0]
