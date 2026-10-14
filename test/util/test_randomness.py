import random

import numpy as np

from nlpaug.util.selection.randomness import Randomness


def test_randomness_seed_is_deterministic():
    Randomness.seed(123)
    first_random = random.random()
    first_numpy = np.random.rand()

    Randomness.seed(123)
    assert random.random() == first_random
    assert np.random.rand() == first_numpy
