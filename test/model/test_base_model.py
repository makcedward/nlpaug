from nlpaug.model.base_model import Model


def test_model_sample_supports_lists_and_int_ranges():
    sampled = Model.sample(['a', 'b', 'c'], num=2)
    assert len(sampled) == 2
    assert set(sampled).issubset({'a', 'b', 'c'})

    sampled_ints = Model.sample(5, num=3)
    assert len(sampled_ints) == 3
    assert all(0 <= value < 5 for value in sampled_ints)
