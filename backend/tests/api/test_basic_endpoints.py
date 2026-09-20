"""Endpoints that need no model, no auth, and no external state."""


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "CareNavigator AI is running"


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert "metrics" in body
    assert body["available_models"] == 0


def test_health_check_is_cached(client, api_module):
    first = client.get("/health").json()
    api_module.cache_service._cache["poisoned"] = "value"
    second = client.get("/health").json()
    # Served from the 1s response cache, so it must be byte-identical
    assert first == second


def test_status(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["application"] == "CareNavigator AI"
    assert body["cache_status"]["type"] == "in-memory"


def test_models_empty(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available_models"] == []
    assert body["count"] == 0


def test_cache_stats(client, api_module):
    api_module.cache_service.set_prediction("flu", {"age": 30}, {"prediction": 1})
    resp = client.get("/cache/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_size"] == 1
    assert body["max_cache_size"] == api_module.cache_service._max_size


def test_metrics_reflects_request_counts(client):
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_metrics"]["requests_total"] >= 1


def test_performance_endpoint(client):
    resp = client.get("/performance")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_cache" in body
    assert "cache_stats" in body


def test_debug_metadata_for_missing_model(client):
    resp = client.get("/debug/metadata/nonexistent_disease")
    assert resp.status_code == 200
    body = resp.json()
    assert body["folder_exists"] is False


def test_model_metadata_404_for_unknown_disease(client):
    resp = client.get("/models/nonexistent_disease/metadata")
    assert resp.status_code == 404
