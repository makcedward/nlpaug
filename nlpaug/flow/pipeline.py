from nlpaug import Augmenter
from nlpaug.augmenter.char import CharAugmenter
from nlpaug.util import Method
from nlpaug.util.progress import optional_progress_bar


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

    def augment(self, data, n=1, num_thread=1, progress_bar=False):
        """
        :param data: Data for augmentation
        :param int n: Number of augmented output for a single input. A list input
            returns one result per input item, regardless of n.
        :param int num_thread: Number of thread for data augmentation. Use this option when you are using CPU and
            n is larger than 1
        :param bool progress_bar: Default is False. If True, show a tqdm progress bar
            for the augmentation loop. Requires `tqdm` (`pip install tqdm`); if tqdm
            is not installed, a warning is emitted once and augmentation proceeds
            without a bar.
        :return: Augmented data

        >>> augmented_data = flow.augment(data)
        """

        if isinstance(data, list):
            return self._augment(data, num_thread=num_thread, progress_bar=progress_bar)

        max_retry_times = 3  # max loop times of n to generate expected number of outputs
        results = []
        is_duplicate_fx = self.get_is_duplicate_fx()
        # `data` may be a single item (str/ndarray) or a list of items when
        # flows are chained (see issue #340). Keep a flat reference for
        # duplicate checks without nesting lists-in-lists.
        flat_data = data if isinstance(data, list) else [data]

        for _ in range(max_retry_times+1):
            augmented_results = []
            if num_thread == 1:
                augmented_results = [
                    self._augment(data)
                    for _ in optional_progress_bar(range(n), enable=progress_bar, desc=self.name)
                ]
            else:
                if self.device == 'cpu':
                    augmented_results = self._parallel_augment(self._augment, data, n=n, num_thread=num_thread)
                    
                # TODO: Externalize to util for checking
                elif 'cuda' in self.device:
                    # TODO: support multiprocessing for GPU
                    # https://discuss.pytorch.org/t/using-cuda-multiprocessing-with-single-gpu/7300
                    augmented_results = [
                        self._augment(data)
                        for _ in optional_progress_bar(range(n), enable=progress_bar, desc=self.name)
                    ]
                else:
                    raise ValueError('Unsupported device mode [{}]. Only support `cpu` or `cuda`'.format(self.device))

            # Flatten nested list (each _augment returns a flat list of items)
            flattened = []
            for sub_results in augmented_results:
                if isinstance(sub_results, list):
                    for r in sub_results:
                        if r is None:
                            continue
                        try:
                            is_empty = len(r) == 0
                        except TypeError:
                            is_empty = False
                        if not is_empty:
                            flattened.append(r)
                elif sub_results is not None:
                    flattened.append(sub_results)
            augmented_results = flattened
            for augmented_result in augmented_results:
                if is_duplicate_fx is None:
                    if isinstance(augmented_result, list):
                        results.extend(augmented_result)
                    else:
                        results.append(augmented_result)
                # Only filter outputs identical to the input (not against other
                # outputs): deterministic sub-augmenters must still satisfy
                # n>1 with repeats (see test_pipeline_duplicate_resolution...).
                # Cross-batch dedup lives in _augment; re-filtering here would
                # incorrectly collapse ['a1', 'a1'] to ['a1'].
                elif not is_duplicate_fx(flat_data, augmented_result):
                    if isinstance(augmented_result, list):
                        results.extend(augmented_result)
                    else:
                        results.append(augmented_result)

                if len(results) >= n:
                    break
            if len(results) >= n:
                break

        if len(results) == 0:
            if len(data) == 0:
                return []
            if isinstance(data, list):
                return list(data)
            return [data]
        return results[:n]

    def _augment(self, data, n=1, num_thread=1, progress_bar=False):
        results = []
        augmented_data = data[:]
        parent_include_detail = self.include_detail

        change_logs = []

        for aug in optional_progress_bar(self, enable=progress_bar, desc=self.name):
            if not self.draw():
                continue

            aug.include_detail = parent_include_detail  # Follow parent setting
            aug.parent_change_seq = self.parent_change_seq+len(change_logs)
            augmented_data = aug.augment(augmented_data, n=n, num_thread=num_thread)
            if aug.include_detail:
                # (augmented_data, change_log)
                change_logs.extend(augmented_data[1])
                augmented_data = augmented_data[0]

        # Batch rows must remain aligned, including unchanged, empty and duplicate
        # outputs. The n limit and duplicate filtering only apply to single inputs.
        if isinstance(data, list):
            if parent_include_detail:
                return augmented_data, change_logs
            return augmented_data

        # Normalize to flat lists: sub-augmenters always return lists, so after
        # chaining `augmented_data` is a list of items, not a single item.
        # Keep outputs flat to avoid [['text']] nesting (issue #340).
        if isinstance(augmented_data, list):
            flat_augmented = augmented_data
        else:
            flat_augmented = [augmented_data]
        flat_data = data if isinstance(data, list) else [data]

        # Data format output of each augmenter should be same
        for aug in self:
            if isinstance(aug, Pipeline):
                results.extend(flat_augmented)
                continue
            added = False
            for item in flat_augmented:
                try:
                    is_dup = aug.is_duplicate(results + flat_data, item)
                except Exception:
                    # If duplicate check cannot handle this type, be conservative
                    # and keep the item.
                    is_dup = False
                if not is_dup:
                    results.append(item)
                    added = True
            # Preserve original single-append-then-break semantics: only the
            # first non-duplicate batch is kept.
            if added:
                break

        output = None
        if len(results) == 0:
            # if not result, return itself (keep flat: list in -> list out)
            if isinstance(data, list):
                output = list(data)
            else:
                output = [data]
        else:
            output = results[:n]

        if parent_include_detail:
            return output, change_logs
        return output
