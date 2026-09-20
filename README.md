# CareNavigator AI

Healthcare intelligence platform: a FastAPI backend serving insurance-plan
matching, medical-document summarization, and AutoML disease-risk
prediction, with a Streamlit frontend on top. Built to be genuinely
production-grade — see [Performance & load testing](#-performance--load-testing)
for real AWS numbers, not marketing copy.

---

## Is this demoable?

**Yes**, in about 2 minutes, with a couple of things worth knowing up front:

- **`/insurance-match/` and `/summary` work immediately**, no setup, no trained
  model. They're intentionally fast/simplified — canned summaries for a
  handful of known conditions, a static ranked plan list — not a full NLP
  pipeline. Good for demoing the API shape and speed, not for demoing real
  matching accuracy.
- **`/predict` needs a trained model first.** None ship pre-trained (`backend/models/`
  is empty). Sample datasets and AutoGluon configs *are* included
  (`backend/uploads/heart.csv`, `covid19_binary.csv`, `HepatitisCdata.csv`,
  `small_heart.csv`), so a real end-to-end **upload → train → predict** demo
  is doable — see [Demo walkthrough](#-demo-walkthrough) below. Training is a
  real AutoGluon run, so budget a few minutes even at a low `time_limit`.
- **The AWS production architecture is real, scripted infrastructure-as-code**,
  not a screenshot — `deploy/aws/production/provision.py` stands up an actual
  ALB + auto-scaling ECS Fargate service, and it's been load-tested at 1000+
  concurrent requests on live AWS infra (results below). It's torn down after
  each run to avoid ongoing cost, so there's no permanently-live URL — running
  `provision.py` yourself stands one up in ~15 minutes.

---

## Features

- **Insurance plan matching** (`POST /insurance-match/`) — fast, cached plan
  recommendations from a free-text description.
- **Medical document summarization** (`POST /summary`) — quick condition
  summaries.
- **Disease risk prediction** (`POST /predict`) — AutoGluon-trained tabular
  models, served with response caching and a concurrency-limiting semaphore.
- **No-code AutoML training** (`POST /upload-and-train`) — upload a CSV,
  auto-generate a training config, train a model end-to-end via AutoGluon.
- **Operational endpoints** — `/health`, `/status`, `/models`, `/metrics`,
  `/cache/stats`, `/performance` for monitoring and load-testing.
- **Streamlit frontend** (`backend/app.py`) — a simple UI over the API for
  all of the above.

---

## Tech stack

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/)/[uvloop](https://github.com/MagicStack/uvloop)
- **ML/AutoML:** [AutoGluon](https://auto.gluon.ai/) (tabular)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Packaging:** Docker
- **Cloud deployment:** AWS (ALB + ECS Fargate, auto-scaling) — see `deploy/aws/`
- **Testing:** pytest (`backend/tests/`)

---

## 🖼️ Screenshots

**Insurance Planner**
![Insurance Planner](https://github.com/user-attachments/assets/560887da-7560-42c5-852f-a27eef38270f)

**Disease Predictor**
![Disease Predictor](https://github.com/user-attachments/assets/60c79fb0-b4f4-4b04-bd3a-29f770ed0381)

---

## Quick start

### Docker (recommended — runs backend + frontend together)

```bash
git clone https://github.com/Jwat205/CareNavigator-AI.git
cd CareNavigator-AI
docker build -t carenavigator-ai .
docker run -p 8000:8000 -p 8501:8501 carenavigator-ai
```

- API: http://localhost:8000
- Streamlit UI: http://localhost:8501

### Local (without Docker)

```bash
pip install -r requirements.txt
cd backend
./start.sh    # starts uvicorn on :8000 and Streamlit on :8501
```

`API_KEY` is optional — unset, the admin endpoints (`/upload-and-train`,
`/reload-plans/`, `/cache/clear`, `/update-registry`) run unauthenticated,
which is what you want for local demoing. Set it in production.

---

## Demo walkthrough

**1. Fast endpoints — no setup:**

```bash
curl -X POST http://localhost:8000/insurance-match/ \
  -H "Content-Type: application/json" \
  -d '{"description": "low premium plan for a family of four"}'

curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{"condition_name": "diabetes", "raw_text": "..."}'
```

**2. Train a real model, then predict:**

```bash
curl -X POST http://localhost:8000/upload-and-train \
  -F "file=@backend/uploads/small_heart.csv" \
  -F "time_limit=60"
```

Once training finishes (`GET /models` will list it), predict against it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"disease": "small_heart", "inputs": {"...": "..."}}'
```

(Use `GET /models/small_heart/metadata` to see the exact input fields the
trained model expects.)

---

## Testing

```bash
cd backend
pytest
```

Covers auth, the core REST endpoints, insurance/summary matching, prediction,
and the cache service — see `backend/tests/`.

---

## 📊 Performance & load testing

This isn't a claim taken on faith — it's backed by real load tests against
real infrastructure, with raw logs checked into the repo.

| Claim | Status | Evidence |
|---|---|---|
| Sub-100ms latency, single instance, ≤100 concurrent | ✅ Proven | [`backend/benchmarks/AWS_LOAD_TEST_RESULTS.md`](backend/benchmarks/AWS_LOAD_TEST_RESULTS.md) |
| Zero-error handling of 1000-1500 concurrent requests | ✅ Proven (ALB + Fargate) | [`backend/benchmarks/AWS_LOAD_TEST_RESULTS_PRODUCTION.md`](backend/benchmarks/AWS_LOAD_TEST_RESULTS_PRODUCTION.md) |
| Sub-100ms latency *at* 1000 concurrent | ✅ 6/9 endpoints proven, 3/9 close (82-106ms avg) | same doc, "Follow-up" section |

Short version: a single box fails hard past 500 concurrent connections. The
production architecture (`deploy/aws/production/` — ALB + auto-scaling ECS
Fargate) fixes that: zero errors at every concurrency level up to 1500,
tested from a same-VPC load generator against live AWS infrastructure. With
enough baseline task capacity and a warmed load balancer, most endpoints hold
sub-100ms even at 1000 simultaneous connections.

Every number, raw log, and root-cause writeup (including bugs found and
fixed along the way) is in `backend/benchmarks/`. Nothing there is simulated.

---

## Deploying it yourself

```bash
cd deploy/aws/production
pip install -r ../requirements.txt
python provision.py                 # stands up ALB + auto-scaling ECS Fargate service
python run_production_loadtest.py 1 10 50 100 500 1000   # optional: prove it yourself
python teardown.py                  # tear down everything (do this when done — it bills)
```

See [`deploy/aws/production/README.md`](deploy/aws/production/README.md) for
prerequisites (AWS credentials, Docker, IAM policy) and design rationale.

---

## License

See [LICENSE](LICENSE).
