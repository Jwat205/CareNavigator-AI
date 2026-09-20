"""Shared fixtures for the CareNavigator AI backend test suite.

`api.py` (and the modules it imports) pull in AutoGluon purely to load and run
trained models. None of the endpoints under test here need a real model —
they either avoid the model path entirely or have it mocked — so a
lightweight stub is installed when the real package isn't available. This
keeps the suite runnable without a multi-GB AutoGluon install.
"""
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _ensure_autogluon_stub():
    try:
        import autogluon.tabular  # noqa: F401
        return
    except ImportError:
        pass

    autogluon_pkg = types.ModuleType("autogluon")
    tabular_mod = types.ModuleType("autogluon.tabular")

    class TabularPredictor:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def load(cls, path):
            raise FileNotFoundError(f"stub TabularPredictor: no model at {path}")

        def predict(self, df):
            raise NotImplementedError("stub TabularPredictor.predict")

        def predict_proba(self, df):
            raise NotImplementedError("stub TabularPredictor.predict_proba")

        @property
        def feature_metadata(self):
            raise NotImplementedError("stub TabularPredictor.feature_metadata")

    tabular_mod.TabularPredictor = TabularPredictor
    autogluon_pkg.tabular = tabular_mod
    sys.modules["autogluon"] = autogluon_pkg
    sys.modules["autogluon.tabular"] = tabular_mod


_ensure_autogluon_stub()


@pytest.fixture(autouse=True)
def _chdir_backend(monkeypatch):
    """api.py reads/writes several paths (insurance_plans.json, uploads/,
    configs/) relative to the process cwd, so pin it to backend/ for every
    test regardless of where pytest was invoked from."""
    monkeypatch.chdir(BACKEND_DIR)


@pytest.fixture(scope="session")
def api_module():
    import api  # imports the module under test once for the whole session
    return api


@pytest.fixture
def client(api_module):
    return TestClient(api_module.app)


@pytest.fixture(autouse=True)
def _reset_shared_state(api_module):
    """api.py keeps process-global caches/counters; clear them before every
    test so results don't depend on test execution order."""
    api_module.cache_service._cache.clear()
    api_module.model_cache._models.clear()
    api_module.model_cache._loading_events.clear()
    api_module.metrics.clear()
    api_module._resp_cache._store.clear()
    api_module._meta_cache._store.clear()
    api_module._endpoint_cache._store.clear()
    api_module._models_cache["result"] = None
    api_module._models_cache["ts"] = 0.0
    yield


@pytest.fixture
def with_api_key(api_module, monkeypatch):
    """Enable API-key auth for a test and return the key to send."""
    monkeypatch.setattr(api_module, "API_KEY", "test-secret-key")
    return "test-secret-key"


@pytest.fixture
def without_api_key(api_module, monkeypatch):
    """Explicitly disable API-key auth (the default when API_KEY is unset)."""
    monkeypatch.setattr(api_module, "API_KEY", None)
