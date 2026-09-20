"""/predict delegates to OptimizedModelCache.predict_with_cache in a thread
pool; these tests mock that call so no real AutoGluon model is needed."""
import asyncio

import pytest

from utils import StrictValidationError


@pytest.fixture(autouse=True)
def _predict_semaphore(api_module):
    """Normally created in the FastAPI lifespan handler, which TestClient
    doesn't run unless used as a context manager — set it up directly."""
    api_module._predict_semaphore = asyncio.Semaphore(8)


def test_predict_success(client, api_module, monkeypatch):
    expected = {"prediction": "1", "probabilities": None, "status": "success", "cached": False}
    monkeypatch.setattr(api_module.model_cache, "predict_with_cache", lambda *a, **k: expected)

    resp = client.post("/predict", json={"disease": "diabetes", "inputs": {"age": 45}})

    assert resp.status_code == 200
    assert resp.json() == expected


def test_predict_passes_through_disease_inputs_and_strict_flag(client, api_module, monkeypatch):
    captured = {}

    def fake_predict(model_name, inputs, strict):
        captured["args"] = (model_name, inputs, strict)
        return {"prediction": "0", "status": "success"}

    monkeypatch.setattr(api_module.model_cache, "predict_with_cache", fake_predict)

    client.post("/predict", json={"disease": "diabetes", "inputs": {"age": 45}, "strict": True})

    assert captured["args"] == ("diabetes", {"age": 45}, True)


def test_predict_strict_validation_error_returns_422(client, api_module, monkeypatch):
    def raise_strict(*a, **k):
        raise StrictValidationError(["age", "sex"])

    monkeypatch.setattr(api_module.model_cache, "predict_with_cache", raise_strict)

    resp = client.post("/predict", json={"disease": "diabetes", "inputs": {}, "strict": True})

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["missing_fields"] == ["age", "sex"]


def test_predict_generic_error_returns_500(client, api_module, monkeypatch):
    def raise_error(*a, **k):
        raise ValueError("model exploded")

    monkeypatch.setattr(api_module.model_cache, "predict_with_cache", raise_error)

    resp = client.post("/predict", json={"disease": "diabetes", "inputs": {}})

    assert resp.status_code == 500
    assert "model exploded" in resp.json()["detail"]


def test_predict_returns_429_when_semaphore_exhausted(client, api_module):
    api_module._predict_semaphore = asyncio.Semaphore(0)

    resp = client.post("/predict", json={"disease": "diabetes", "inputs": {}})

    assert resp.status_code == 429
