from unittest.mock import Mock

from nlpaug.util.model_cache import ModelCache


def test_get_returns_none_for_missing_key():
    cache = ModelCache()

    assert cache.get('missing') is None


def test_get_or_create_caches_by_key():
    cache = ModelCache()
    calls = []

    def factory():
        calls.append(1)
        return object()

    first = cache.get_or_create('k', factory)
    second = cache.get_or_create('k', factory)

    assert first is second
    assert len(calls) == 1
    assert cache.get('k') is first


def test_get_or_create_applies_updates_on_cache_hit():
    cache = ModelCache()
    model = Mock()
    cache._models['k'] = model

    result = cache.get_or_create('k', factory=Mock(), updates={'foo': 'bar'})

    assert result is model
    assert model.foo == 'bar'


def test_get_or_create_ignores_none_and_empty_updates():
    cache = ModelCache()
    model = Mock()
    model_with_attr = Mock(spec=[])
    cache._models['a'] = model
    cache._models['b'] = model_with_attr

    assert cache.get_or_create('a', factory=Mock(), updates=None) is model
    assert cache.get_or_create('b', factory=Mock(), updates={}) is model_with_attr


def test_get_or_create_force_reload_replaces_entry():
    cache = ModelCache()
    old, new = object(), object()

    cache._models['k'] = old
    result = cache.get_or_create('k', factory=lambda: new, force_reload=True)

    assert result is new
    assert cache.get('k') is new


def test_clear_empties_cache():
    cache = ModelCache()
    cache._models['k'] = object()

    cache.clear()

    assert cache.get('k') is None
