from nlpaug import Augmenter
from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Method


class Pipeline(Augmenter, list):
    def __init__(self, action, name='Pipeline', aug_min=1, aug_p=1, flow=None, verbose=0):
        Augmenter.__init__(self, name=name, method=Method.FLOW,
                           action=action, aug_min=aug_min, verbose=verbose)
        self.aug_p = aug_p
        if flow is None:
            list.__init__(self, [])
        elif isinstance(flow, (Augmenter, CharAugmenter)):
            list.__init__(self, [flow])
        elif isinstance(flow, list):
            for subflow in flow:
                if not isinstance(subflow, Augmenter):
                    raise ValueError('At least one of the flow does not belongs to Augmenter')
            list.__init__(self, flow)
        else:
            raise Exception(
                'Expected None, Augmenter or list of Augmenter while {} is passed'.format(
                    type(flow)))

    def draw(self):
        raise NotImplementedError

    def augment(self, data, n=1):
        """
        :param data: Data for augmentation
        :param int n: Number of augmented output
        :return: Augmented data

        >>> augmented_data = flow.augment(data)
        """
        results = [data]
        for _ in range(n*5):
            augmented_data = data[:]
            for aug in self:
                if not self.draw():
                    continue
                augmented_data = aug.augment(augmented_data, n=1)

            # Data format output of each augmenter should be same
            for aug in self:
                if aug.__class__.__bases__[0] == Pipeline:
                    results.append(augmented_data)
                    continue
                if not aug.is_duplicate(results, augmented_data):
                    results.append(augmented_data)
                break

            if len(results) >= n+1:
                break

        # only have input data
        if len(results) == 1:
            if n == 1:
                return results[0]
            else:
                return [results[0]]

        # return 1 record as n == 1
        if n == 1 and len(results) >= 2:
            return results[1]

        # return all records
        return results[1:]