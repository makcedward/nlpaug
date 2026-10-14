import math
import random
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

from nlpaug.util import Action, Method, WarningException, WarningName, WarningCode, WarningMessage


class Augmenter:
    ACTION_METHOD_NAMES = {
        Action.INSERT: 'insert',
        Action.SUBSTITUTE: 'substitute',
        Action.SWAP: 'swap',
        Action.DELETE: 'delete',
        Action.CROP: 'crop',
        Action.SPLIT: 'split',
    }
    ONE_TO_MANY_AUGMENTERS = {'LambadaAug'}
    MODEL_BATCH_AUGMENTERS = {
        'AbstSummAug',
        'BackTranslationAug',
        'ContextualWordEmbsAug',
        'ContextualWordEmbsForSentenceAug',
    }

    def __init__(self, name, method, action, aug_min, aug_max, aug_p=0.1, device='cpu', 
        include_detail=False, verbose=0):

        self.name = name
        self.action = action
        self.method = method
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.aug_p = aug_p
        self.device = device
        self.verbose = verbose
        self.include_detail = include_detail

        self.parent_change_seq = 0

        self._validate_augmenter(method, action)

    @classmethod
    def _validate_augmenter(cls, method, action):
        if method not in Method.getall():
            raise ValueError(
                'Method must be one of {} while {} is passed'.format(Method.getall(), method))

        if action not in Action.getall():
            raise ValueError(
                'Action must be one of {} while {} is passed'.format(Action.getall(), action))

    def augment(self, data, n=1, num_thread=1):
        """
        :param object/list data: Data for augmentation. It can be list of data (e.g. list 
            of string or numpy) or single element (e.g. string or numpy). Numpy format only
            supports audio or spectrogram data. For text data, only support string or
            list of string.
        :param int n: Default is 1. Number of unique augmented output. Will be force to 1 
            if input is list of data
        :param int num_thread: Number of thread for data augmentation. Use this option 
            when you are using CPU and n is larger than 1
        :return: Augmented data

        >>> augmented_data = aug.augment(data)

        """
        max_retry_times = 3  # max loop times of n to generate expected number of outputs
        aug_num = 1 if isinstance(data, list) else n
        expected_output_num = len(data) if isinstance(data, list) else aug_num

        exceptions = self._validate_augment(data)
        # TODO: Handle multiple exceptions
        for exception in exceptions:
            if isinstance(exception, WarningException):
                if self.verbose > 0:
                    exception.output()

                # Return empty value per data type
                if isinstance(data, str):
                    return []
                elif isinstance(data, list):
                    return []
                elif isinstance(data, np.ndarray):
                    return np.array([])

                return []

        clean_data = self.clean(data)
        action_fx = self._get_action_handler()

        for _ in range(max_retry_times+1):
            augmented_results = []
            augmenter_name = self.__class__.__name__

            # By design, it is one-to-many
            if augmenter_name in self.ONE_TO_MANY_AUGMENTERS:
                augmented_results = action_fx(clean_data, n=n)
            # PyTorch's augmenter
            elif augmenter_name in self.MODEL_BATCH_AUGMENTERS:
                for _ in range(aug_num):
                    result = action_fx(clean_data)
                    if isinstance(result, list):
                        augmented_results.extend(result)
                    else:
                        augmented_results.append(result)
            # Multi inputs
            elif isinstance(data, list):
                # Single Thread
                if num_thread == 1:
                    augmented_results = [action_fx(d) for d in clean_data]

                # Multi Thread
                else:
                    batch_data = [clean_data[i:i+num_thread] for i in range(0, len(clean_data), num_thread)]
                    for mini_batch_data in batch_data:
                        threaded_results = self._parallel_augments(
                            lambda item: action_fx(item),
                            mini_batch_data,
                        )
                        augmented_results.extend(threaded_results)

            # Single input with/without multiple input
            else:
                # Single Thread
                if num_thread == 1:
                    augmented_results = [action_fx(clean_data) for _ in range(n)]

                # Multi Thread
                else:
                    augmented_results = self._parallel_augment(action_fx, clean_data, n=n, num_thread=num_thread)

            if len(augmented_results) >= expected_output_num:
                break

        if len(augmented_results) == 0:
            # if not result, return itself
            if isinstance(data, list):
                return data
            # Single input with/without multiple input
            else:
                return [data]

        if isinstance(augmented_results, pd.DataFrame):
            return augmented_results
        else:
            if isinstance(data, list):
                return augmented_results
            else:
                return augmented_results[:n]

        # return augmented_results

    # def augments(self, data, num_thread=1):
    #     """
    #     :param list data: List of data
    #     :param int num_thread: Number of thread for data augmentation. Use this option when you are using CPU and
    #         n is larger than 1. Do NOT support GPU process.
    #     :return: Augmented data (Does not follow original order)

    #     >>> augmented_data = aug.augment(data)

    #     """
    #     n = 1
    #     augmented_results = []
    #     if num_thread == 1 or self.device == 'cuda':
    #         for d in data:
    #             augmented_result = self.augment(data=d, n=n, num_thread=1)  # TOOD: cuda does not support mulithread
    #             if n == 1:
    #                 augmented_results.append(augmented_result)
    #             else:
    #                 augmented_results.extend(augmented_result)
    #     else:
    #         batch_data = [data[i:i+num_thread] for i in range(0, len(data), num_thread)]
    #         for i in range(n):
    #             for mini_batch_data in batch_data:
    #                 augmented_results.extend(self._parallel_augments(self.augment, mini_batch_data))

    #     return augmented_results

    @classmethod
    def _validate_augment(cls, data):
        if data is None or len(data) == 0:
            return [WarningException(name=WarningName.INPUT_VALIDATION_WARNING,
                                     code=WarningCode.WARNING_CODE_001, msg=WarningMessage.LENGTH_IS_ZERO)]

        return []

    @classmethod
    def _parallel_augment(cls, action_fx, data, n, num_thread=2):
        if n <= 1:
            return [action_fx(data)]

        with ThreadPool(num_thread) as pool:
            return pool.map(action_fx, [data] * n)

    @classmethod
    def _parallel_augments(cls, action_fx, data):
        if len(data) <= 1:
            return [action_fx(data[0])] if data else []

        with ThreadPool(len(data)) as pool:
            return pool.map(action_fx, data)

    def _get_action_handler(self):
        method_name = self.ACTION_METHOD_NAMES.get(self.action)
        if method_name is None:
            raise ValueError('Unsupported action {}'.format(self.action))
        return getattr(self, method_name)

    def insert(self, data):
        raise NotImplementedError

    def substitute(self, data):
        raise NotImplementedError

    def swap(self, data):
        raise NotImplementedError

    def delete(self, data):
        raise NotImplementedError

    def crop(self, data):
        raise NotImplementedError        

    def split(self, data):
        raise NotImplementedError

    def tokenizer(self, tokens):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    @classmethod
    def is_duplicate(cls, dataset, data):
        raise NotImplementedError

    @classmethod
    def prob(cls):
        return np.random.random()

    @classmethod
    def sample(cls, x, num=None):
        if isinstance(x, list):
            return random.sample(x, num)
        elif isinstance(x, int):
            return np.random.randint(1, x-1)

    @classmethod
    def clean(cls, data):
        raise NotImplementedError

    def _generate_aug_cnt(self, size, aug_min, aug_max, aug_p=None):
        if aug_p is not None:
            percent = aug_p
        elif self.aug_p:
            percent = self.aug_p
        else:
            percent = 0.3
        cnt = int(math.ceil(percent * size))

        if aug_min and cnt < aug_min:
            return aug_min
        if aug_max and cnt > aug_max:
            return aug_max
        return cnt

    def generate_aug_cnt(self, size, aug_p=None):
        if size == 0:
            return 0
        return self._generate_aug_cnt(size, self.aug_min, self.aug_max, aug_p)

    def generate_aug_idxes(self, inputs):
        aug_cnt = self.generate_aug_cnt(len(inputs))
        token_idxes = [i for i, _ in enumerate(inputs)]
        aug_idxes = self.sample(token_idxes, aug_cnt)
        return aug_idxes

    def _get_random_aug_idxes(self, data):
        aug_cnt = self.generate_aug_cnt(len(data))
        idxes = self.pre_skip_aug(data)
        if len(idxes) < aug_cnt:
            aug_cnt = len(idxes)

        aug_idxes = self.sample(idxes, aug_cnt)

        return aug_idxes

    def __str__(self):
        return 'Name:{}, Action:{}, Method:{}'.format(self.name, self.action, self.method)
