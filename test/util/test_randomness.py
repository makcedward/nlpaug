import random

import numpy as np
import pytest

from nlpaug.util.selection.randomness import Randomness


def test_randomness_seed_is_deterministic():
    Randomness.seed(123)
    first_random = random.random()
    first_numpy = np.random.rand()

    Randomness.seed(123)
    assert random.random() == first_random
    assert np.random.rand() == first_numpy


def test_randomness_seed_covers_torch_branch():
    torch = pytest.importorskip('torch')

    Randomness.seed(123)
    first = torch.rand(3)

    Randomness.seed(123)
    second = torch.rand(3)

    assert torch.equal(first, second)
