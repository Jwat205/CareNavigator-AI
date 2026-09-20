import time

from api import SimpleCache, ResponseCache


def test_simple_cache_prediction_roundtrip():
    cache = SimpleCache()
    cache.set_prediction("flu", {"age": 30}, {"prediction": 1})
    assert cache.get_prediction("flu", {"age": 30}) == {"prediction": 1}


def test_simple_cache_prediction_miss_for_different_features():
    cache = SimpleCache()
    cache.set_prediction("flu", {"age": 30}, {"prediction": 1})
    assert cache.get_prediction("flu", {"age": 31}) is None


def test_simple_cache_key_ignores_dict_key_order():
    cache = SimpleCache()
    cache.set_prediction("flu", {"age": 30, "sex": 1}, {"prediction": 1})
    assert cache.get_prediction("flu", {"sex": 1, "age": 30}) == {"prediction": 1}


def test_simple_cache_summary_roundtrip():
    cache = SimpleCache()
    cache.set_summary("diabetes", "hash123", "a short summary")
    assert cache.get_summary("diabetes", "hash123") == "a short summary"


def test_simple_cache_evicts_oldest_when_full():
    cache = SimpleCache()
    cache._max_size = 2
    cache.set_prediction("a", {"x": 1}, "first")
    cache.set_prediction("b", {"x": 2}, "second")
    cache.set_prediction("c", {"x": 3}, "third")

    assert len(cache._cache) == 2
    assert cache.get_prediction("a", {"x": 1}) is None
    assert cache.get_prediction("c", {"x": 3}) == "third"


def test_response_cache_returns_value_within_ttl():
    cache = ResponseCache(ttl=1.0)
    cache.set("key", {"value": 42})
    assert cache.get("key") == {"value": 42}


def test_response_cache_expires_after_ttl():
    cache = ResponseCache(ttl=0.05)
    cache.set("key", {"value": 42})
    time.sleep(0.1)
    assert cache.get("key") is None


def test_response_cache_miss_for_unknown_key():
    cache = ResponseCache(ttl=1.0)
    assert cache.get("missing") is None
