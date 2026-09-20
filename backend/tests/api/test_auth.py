"""State-changing endpoints require X-API-Key only once API_KEY is configured
(see require_api_key in api.py) — unset API_KEY is a deliberate dev-mode
bypass, not a bug."""
import pytest

PROTECTED_POST_ENDPOINTS = [
    "/reload-plans/",
    "/update-registry",
    "/cache/clear",
    "/admin/maintenance/run",
]


@pytest.mark.parametrize("path", PROTECTED_POST_ENDPOINTS)
def test_protected_endpoint_rejects_missing_key(client, with_api_key, path):
    resp = client.post(path)
    assert resp.status_code == 403


@pytest.mark.parametrize("path", PROTECTED_POST_ENDPOINTS)
def test_protected_endpoint_rejects_wrong_key(client, with_api_key, path):
    resp = client.post(path, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 403


def test_reload_plans_succeeds_with_correct_key(client, with_api_key):
    resp = client.post("/reload-plans/", headers={"X-API-Key": with_api_key})
    assert resp.status_code == 200
    assert resp.json()["message"] == "Insurance plans reloaded successfully"


def test_cache_clear_succeeds_with_correct_key(client, with_api_key, api_module):
    api_module.cache_service.set_prediction("flu", {"age": 1}, {"prediction": 1})
    resp = client.post("/cache/clear", headers={"X-API-Key": with_api_key})
    assert resp.status_code == 200
    assert resp.json()["cache_size"] == 0


def test_update_registry_succeeds_with_correct_key(client, with_api_key):
    resp = client.post("/update-registry", headers={"X-API-Key": with_api_key})
    assert resp.status_code == 200
    assert resp.json()["message"] == "Model registry updated successfully"


@pytest.mark.parametrize("path", PROTECTED_POST_ENDPOINTS)
def test_protected_endpoint_open_when_api_key_unset(client, without_api_key, path):
    resp = client.post(path)
    assert resp.status_code != 403
