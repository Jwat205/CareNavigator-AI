"""The /insurance-match/ and /summary endpoints are intentionally
hard-coded/fast-path implementations (see comments in api.py) rather than
using EnhancedInsuranceMatcher — tests reflect that actual behavior."""


def test_insurance_match_returns_fixed_plans(client):
    resp = client.post("/insurance-match/", json={"description": "45 year old with diabetes in Texas"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["matched_plans"] == ["FastCare Basic", "QuickHealth Pro", "SpeedInsure Plus"]
    assert body["fast_mode"] is True
    assert len(body["detailed_matches"]) == 3


def test_insurance_match_requires_description_field(client):
    resp = client.post("/insurance-match/", json={})
    assert resp.status_code == 422


def test_summary_known_condition(client):
    resp = client.post("/summary", json={"condition_name": "diabetes", "raw_text": "irrelevant"})
    assert resp.status_code == 200
    body = resp.json()
    assert "blood sugar" in body["summary"]
    assert body["cached"] is False


def test_summary_repeated_identical_request_is_served_from_cache(client):
    payload = {"condition_name": "diabetes", "raw_text": "irrelevant"}
    first = client.post("/summary", json=payload).json()
    second = client.post("/summary", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["summary"] == first["summary"]


def test_insurance_match_repeated_identical_request_is_served_from_cache(client):
    payload = {"description": "45 year old with diabetes in Texas"}
    first = client.post("/insurance-match/", json=payload).json()
    second = client.post("/insurance-match/", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["matched_plans"] == first["matched_plans"]


def test_summary_unknown_condition_falls_back_to_default(client):
    resp = client.post("/summary", json={"condition_name": "made_up_condition", "raw_text": "irrelevant"})
    assert resp.status_code == 200
    assert resp.json()["summary"] == (
        "This is a medical condition that requires professional healthcare attention and management."
    )


def test_summary_condition_lookup_is_case_insensitive(client):
    resp = client.post("/summary", json={"condition_name": "DIABETES", "raw_text": "x"})
    assert resp.status_code == 200
    assert "blood sugar" in resp.json()["summary"]
