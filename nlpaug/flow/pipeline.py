from nlpaug import Augmenter
from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Method


class Pipeline(Augmenter, list):
    def __init__(self, action, name='Pipeline', aug_p=1, flow=None, include_detail=False, verbose=0):
        Augmenter.__init__(self, name=name, method=Method.FLOW,
                           action=action, aug_min=None, aug_max=None, verbose=verbose, include_detail=include_detail)
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

    def get_is_duplicate_fx(self):
        # Assume all augmenters share same is_duplicate function.
        for aug in self:
            if isinstance(aug, list):
                is_duplicate_fx = aug.get_is_duplicate_fx()
                if is_duplicate_fx is not None:
                    return is_duplicate_fx
            else:
                return aug.is_duplicate

        return None

    def augment(self, data, n=1, num_thread=1):
        """
        :param data: Data for augmentation
        :param int n: Number of augmented output
        :param int num_thread: Number of thread for data augmentation. Use this option when you are using CPU and
            n is larger than 1
        :return: Augmented data

        >>> augmented_data = flow.augment(data)
        """

        max_retry_times = 3  # max loop times of n to generate expected number of outputs
        results = []
        is_duplicate_fx = self.get_is_duplicate_fx()

        for _ in range(max_retry_times+1):
            augmented_results = []
            if num_thread == 1:
                augmented_results = [self._augment(data) for _ in range(n)]
            else:
                if self.device == 'cpu':
                    augmented_results = self._parallel_augment(self._augment, data, n=n, num_thread=num_thread)
                # TODO: Externalize to util for checking
                elif 'cuda' in self.device:
                    # TODO: support multiprocessing for GPU
                    # https://discuss.pytorch.org/t/using-cuda-multiprocessing-with-single-gpu/7300
                    augmented_results = [self._augment(data) for _ in range(n)]
                else:
                    raise ValueError('Unsupported device mode [{}]. Only support `cpu` or `cuda`'.format(self.device))

            for augmented_result in augmented_results:
                if is_duplicate_fx is not None and not is_duplicate_fx(results + [data], augmented_result):
                    results.append(augmented_result)

                if len(results) >= n:
                    break
            if len(results) >= n:
                break

        # TODO: standardize output to list even though n=1
        if len(results) == 0:
            # if not result, return itself
            if n == 1:
                return data
            else:
                return [data]
        if n == 1:
            return results[0]
        return results[:n]

    def _augment(self, data, n=1, num_thread=1):
        results = []
        augmented_data = data[:]
        parent_include_detail = self.include_detail

        change_logs = []

        for aug in self:
            if not self.draw():
                continue

            aug.include_detail = parent_include_detail  # Follow parent setting
            aug.parent_change_seq = self.parent_change_seq+len(change_logs)
            augmented_data = aug.augment(augmented_data, n=n, num_thread=num_thread)
            if aug.include_detail:
                # (augmented_data, change_log)
                change_logs.extend(augmented_data[1])
                augmented_data = augmented_data[0]

        # Data format output of each augmenter should be same
        for aug in self:
            if str(aug.__class__.__bases__[0]) == str(Pipeline):
                results.append(augmented_data)
                continue
            if not aug.is_duplicate(results + [data], augmented_data):
                results.append(augmented_data)
            break

        # TODO: standardize output to list even though n=1
        output = None
        if len(results) == 0:
            # if not result, return itself
            if n == 1:
                output = data
            else:
                output = [data]
        elif n == 1:
            output = results[0]
        else:
            output = results[:n]

        if parent_include_detail:
            return output, change_logs
        return output
