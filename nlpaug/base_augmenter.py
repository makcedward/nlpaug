import random
import numpy as np

from nlpaug.util import Action, Method, WarningException, WarningName, WarningCode, WarningMessage


class Augmenter:
    def __init__(self, name, method, action, aug_min, aug_p=0.1, verbose=0):
        self.name = name
        self.action = action
        self.method = method
        self.aug_min = aug_min
        self.aug_p = aug_p
        self.verbose = verbose

        self.augments = []

        self._validate_augmenter(method, action)

    @classmethod
    def _validate_augmenter(cls, method, action):
        if method not in Method.getall():
            raise ValueError(
                'Method must be one of {} while {} is passed'.format(Method.getall(), method))

        if action not in Action.getall():
            raise ValueError(
                'Action must be one of {} while {} is passed'.format(Action.getall(), action))

    def augment(self, data, n=1):
        """
        :param object data: Data for augmentation
        :param int n: Number of augmented output
        :return: Augmented data

        >>> augmented_data = aug.augment(data)

        """
        exceptions = self._validate_augment(data)
        # TODO: Handle multiple exceptions
        for exception in exceptions:
            if isinstance(exception, WarningException):
                if self.verbose > 0:
                    exception.output()

                # Return empty value per data type
                if isinstance(data, str):
                    return ''
                elif isinstance(data, list):
                    return []
                elif isinstance(data, np.ndarray):
                    return np.array([])

                return None

        results = [data]
        for _ in range(n*5):
            if self.action == Action.INSERT:
                result = self.insert(data)
            elif self.action == Action.SUBSTITUTE:
                result = self.substitute(data)
            elif self.action == Action.SWAP:
                result = self.swap(data)
            elif self.action == Action.DELETE:
                result = self.delete(data)

            if not self.is_duplicate(results, result):
                results.append(result)

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

    @classmethod
    def _validate_augment(cls, data):
        if data is None or len(data) == 0:
            return [WarningException(name=WarningName.INPUT_VALIDATION_WARNING,
                                     code=WarningCode.WARNING_CODE_001, msg=WarningMessage.LENGTH_IS_ZERO)]

        return []

    def insert(self, data):
        raise NotImplementedError()

    def substitute(self, data):
        raise NotImplementedError()

    def swap(self, data):
        raise NotImplementedError()

    def delete(self, data):
        raise NotImplementedError()

    def tokenizer(self, tokens):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    @classmethod
    def is_duplicate(cls, dataset, data):
        raise NotImplementedError

    @classmethod
    def prob(cls):
        return random.random()

    @classmethod
    def sample(cls, x, num):
        return random.sample(x, num)

    def generate_aug_cnt(self, size, aug_p=None):
        if aug_p is not None:
            percent = aug_p
        elif self.aug_p is not None:
            percent = self.aug_p
        else:
            percent = 0.3
        cnt = int(percent * size)
        return cnt if cnt > self.aug_min else self.aug_min

    def generate_aug_idxes(self, inputs):
        aug_cnt = self.generate_aug_cnt(len(inputs))
        token_idxes = [i for i, _ in enumerate(inputs)]
        aug_idxes = self.sample(token_idxes, aug_cnt)
        return aug_idxes

    def __str__(self):
        return 'Name:{}, Action:{}, Method:{}'.format(self.name, self.action, self.method)
