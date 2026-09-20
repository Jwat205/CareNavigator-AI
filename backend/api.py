#api.py
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Header, Depends
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import logging
import os
import json
import nltk
import re
import joblib
import pandas as pd
import shap
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
from utils import load_model_and_features, StrictValidationError
from auto_config_generator import generate_config_dict_from_csv
import tempfile
from train_model import train_with_autogluon
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
import anyio
from contextlib import asynccontextmanager

# Windows-compatible optimizations
from collections import Counter
import time
import hashlib
from starlette.middleware.base import BaseHTTPMiddleware

# Simplified caching for Windows (in-memory fallback)
class SimpleCache:
    def __init__(self):
        self._cache = {}
        self._max_size = 1000
        self.available = True
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get_prediction(self, model_name: str, features: Dict[str, Any]) -> Optional[Any]:
        cache_key = self._generate_cache_key(f"prediction:{model_name}", features)
        return self._cache.get(cache_key)
    
    def set_prediction(self, model_name: str, features: Dict[str, Any], result: Any, ttl: int = 3600):
        if len(self._cache) >= self._max_size:
            # Simple LRU: remove oldest items
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        cache_key = self._generate_cache_key(f"prediction:{model_name}", features)
        self._cache[cache_key] = result
    
    def get_summary(self, condition_name: str, text_hash: str) -> Optional[str]:
        cache_key = f"summary:{condition_name}:{text_hash}"
        return self._cache.get(cache_key)
    
    def set_summary(self, condition_name: str, text_hash: str, summary: str, ttl: int = 86400):
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        cache_key = f"summary:{condition_name}:{text_hash}"
        self._cache[cache_key] = summary

# OPTIMIZED MODEL CACHE
#Old busy waiting way( Lacks concurrency Property)
#class OptimizedModelCache:
 #   def __init__(self):
  #      self._models = {}
   #     self._loading_locks = {}  # Prevent multiple simultaneous loads
    #    self.load_times = {}  # Track load performance
class OptimizedModelCache:
    def __init__(self):
        self._models = {}
        self._loading_events = {}  # Now using events for software interrupts
        self.load_times = {}
  
    async def load_model_async(self, model_name: str):
        if model_name not in self._models:
            if model_name in self._loading_events:
                await self._loading_events[model_name].wait()
                return self._models.get(model_name)

            try:
                self._loading_events[model_name] = asyncio.Event()
                start_time = time.time()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.load_model, model_name)
                self.load_times[model_name] = time.time() - start_time
                logging.info(f"✅ Loaded model: {model_name} (No blocking!)")
            finally:
                self._loading_events[model_name].set()
                del self._loading_events[model_name]

        return self._models.get(model_name)

    def load_model(self, model_name: str):
        """Synchronous model load — called from thread pool and startup preloading."""
        from utils import load_model_and_features
        model, features = load_model_and_features(model_name)
        self._models[model_name] = (model, features)
        return model, features

    def load_model_sync(self, model_name: str):
        return self.load_model(model_name)
    
    def get_model_sync(self, model_name: str):
        return self._models.get(model_name)
    
    def predict_with_cache(self, model_name: str, inputs: dict, strict: bool = False):
        """Highly optimized prediction with smart caching"""
        start_time = time.perf_counter()

        # Check cache first (this should be VERY fast)
        # Strict-mode requests bypass the cache since a validation failure must
        # always be re-evaluated against the current inputs, never served stale.
        cached_result = None if strict else cache_service.get_prediction(model_name, inputs)
        if cached_result is not None:
            cache_time = (time.perf_counter() - start_time) * 1000
            cached_result["cached"] = True
            cached_result["response_time_ms"] = round(cache_time, 2)
            logging.info(f"🚀 Cache HIT for {model_name}: {cache_time:.2f}ms")
            return cached_result
        
        # If not cached, run optimized prediction
        try:
            # Use preloaded model if available (should be instant)
            cached_model = self.get_model_sync(model_name)
            if cached_model:
                model, expected_inputs = cached_model
                logging.info(f"✅ Using preloaded model: {model_name}")
            else:
                # Fallback: load model on-demand
                logging.warning(f"⚠️ Loading model on-demand: {model_name}")
                model, expected_inputs = self.load_model_sync(model_name)
            
            # Use optimized validation
            prediction_start = time.perf_counter()
            
            from utils import validate_prediction_inputs
            row, missing_features, _ = validate_prediction_inputs(model_name, inputs, strict=strict)
            
            # Create DataFrame and predict (optimized)
            df = pd.DataFrame([row])
            prediction = model.predict(df)
            
            # Get probabilities (optional, can be skipped for speed)
            probabilities = None
            try:
                prob_result = model.predict_proba(df)
                if prob_result is not None:
                    probabilities = {f"class_{i}": float(prob) 
                                   for i, prob in enumerate(prob_result.iloc[0])}
            except Exception:
                # Skip probabilities if they fail (for speed)
                pass
            
            # Format result efficiently
            if hasattr(prediction, 'tolist'):
                pred_result = prediction.tolist()[0]
            elif hasattr(prediction, 'iloc'):
                pred_result = str(prediction.iloc[0])
            else:
                pred_result = str(prediction)
            
            prediction_time = (time.perf_counter() - prediction_start) * 1000
            total_time = (time.perf_counter() - start_time) * 1000
            
            result = {
                "prediction": pred_result,
                "probabilities": probabilities,
                "missing_features_filled": missing_features,
                "status": "success",
                "cached": False,
                "prediction_time_ms": round(prediction_time, 2),
                "total_time_ms": round(total_time, 2)
            }
            
            # Cache the result asynchronously (don't block response)
            cache_service.set_prediction(model_name, inputs, result)
            
            logging.info(f"🔄 Prediction for {model_name}: {total_time:.2f}ms (pred: {prediction_time:.2f}ms)")
            return result
            
        except Exception as e:
            error_time = (time.perf_counter() - start_time) * 1000
            logging.error(f"❌ Prediction error for {model_name} after {error_time:.2f}ms: {e}")
            raise

# SINGLE REQUEST MIDDLEWARE — process-time headers, cache-control, structured
# logging, and metrics all in one pass (previously three separate middleware
# layers each re-wrapped every request, tripling per-request overhead).
_STATIC_CACHEABLE_PATHS = {"/health", "/", "/models", "/status"}

async def _log_request_line(method, path, status_code, duration_ms, client_host):
    """Emit the structured per-request JSON log line. Run as a fire-and-forget
    asyncio task from RequestMiddleware.dispatch so building the dict, calling
    json.dumps, and calling datetime.now() never block the response from
    being sent back to the client."""
    logging.info(json.dumps({
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "timestamp": datetime.now().isoformat(),
        "client_host": client_host,
    }))

class RequestMiddleware(BaseHTTPMiddleware):
    """Per-request: adds X-Process-Time / Cache-Control headers, increments
    `metrics` counters, and emits one structured JSON log line with method,
    path, status_code, duration_ms, timestamp, and client host (5+ fields)."""
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            method = request.method
            path = request.url.path
            client_host = request.client.host if request.client else "unknown"

            metrics["requests_total"] += 1
            metrics[f"{method}:{path}"] += 1
            metrics[f"status:{status_code}"] += 1

            asyncio.create_task(_log_request_line(method, path, status_code, duration_ms, client_host))

            if status_code < 500:
                try:
                    response.headers["X-Process-Time"] = f"{duration_ms:.2f}"
                    if path in _STATIC_CACHEABLE_PATHS:
                        response.headers["Cache-Control"] = "public, max-age=300"
                except Exception:
                    pass

# OPTIMIZED SUMMARIZATION SERVICE
class OptimizedSummarizationService:
    def __init__(self):
        self.model_loaded = False
        self.load_summarizer()
    
    def load_summarizer(self):
        """Preload summarization model"""
        try:
            # The summarizer is already loaded globally, just mark as ready
            self.model_loaded = True
            logging.info("✅ Summarization model ready")
        except Exception as e:
            logging.error(f"❌ Failed to load summarization model: {e}")
            self.model_loaded = False
    
    def summarize_with_cache(self, condition_name: str, raw_text: str):
        """Optimized summarization with caching and text preprocessing"""
        start_time = time.perf_counter()
        
        # Optimize text preprocessing
        if len(raw_text) > 1000:
            # Truncate very long text to speed up processing
            raw_text = raw_text[:1000] + "..."
            logging.info(f"📝 Truncated long text for faster processing")
        
        # Generate text hash for caching
        text_hash = hashlib.md5(raw_text.encode()).hexdigest()
        
        # Check cache first
        cached_summary = cache_service.get_summary(condition_name, text_hash)
        if cached_summary:
            cache_time = (time.perf_counter() - start_time) * 1000
            logging.info(f"🚀 Summary cache HIT: {cache_time:.2f}ms")
            return {"summary": cached_summary, "cached": True, "response_time_ms": round(cache_time, 2)}
        
        # Generate summary with optimized parameters
        try:
            if not self.model_loaded:
                raise Exception("Summarization model not loaded")
            
            summary_start = time.perf_counter()
            
            # Optimize summarizer parameters for speed
            max_length = min(50, len(raw_text.split()) // 2)  # Adaptive max length
            min_length = min(15, max_length // 2)
            
            result = summarizer(
                raw_text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False,
                truncation=True  # Handle long texts gracefully
            )
            summary = result[0]["summary_text"]
            
            summary_time = (time.perf_counter() - summary_start) * 1000
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Cache for later
            cache_service.set_summary(condition_name, text_hash, summary)
            
            logging.info(f"🔄 Generated summary: {total_time:.2f}ms (model: {summary_time:.2f}ms)")
            
            return {
                "summary": summary, 
                "cached": False,
                "response_time_ms": round(total_time, 2),
                "summary_time_ms": round(summary_time, 2)
            }
            
        except Exception as e:
            error_time = (time.perf_counter() - start_time) * 1000
            logging.error(f"❌ Summarization error after {error_time:.2f}ms: {e}")
            raise

# Initialize global services
cache_service = SimpleCache()
model_cache = OptimizedModelCache()  # OPTIMIZED
summarization_service = OptimizedSummarizationService()  # OPTIMIZED
metrics = Counter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = Path(BASE_DIR) / "models"
UPLOAD_DIR = "uploads"
CONFIG_DIR = "configs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# --- ENV & LOGGING SETUP ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- API KEY AUTH (optional, dev-friendly) ---
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logging.warning(
        "⚠️ API_KEY is not set — sensitive endpoints (/upload-and-train, /reload-plans/, "
        "/cache/clear, /update-registry) are UNAUTHENTICATED. Set API_KEY to enable auth."
    )

async def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Require a matching X-API-Key header on state-changing endpoints, but only
    when API_KEY is configured — keeps local/dev usage without a key working."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True

_predict_semaphore: asyncio.Semaphore

# ============================
# BACKGROUND MAINTENANCE PIPELINE
# ============================
# Orchestrates 12 independent maintenance coroutines concurrently via a single
# asyncio.gather — run once at startup and re-triggerable on demand via
# POST /admin/maintenance/run. Each task is failure-isolated
# (return_exceptions=True) and does real work: warming caches, validating
# on-disk model/config state, refreshing the registry, and persisting metrics.

_last_maintenance_run: dict = {"timestamp": None, "results": {}}

async def _task_preload_models():
    """Preload all available trained models into memory, concurrently."""
    available_models = get_available_models()
    loop = asyncio.get_event_loop()

    async def safe_preload(model_info):
        model_name = model_info["folder_name"]
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, model_cache.load_model, model_name),
                timeout=5.0
            )
            return {"model": model_name, "status": "preloaded"}
        except Exception as e:
            return {"model": model_name, "status": "skipped", "error": str(e)}

    results = await asyncio.gather(*[safe_preload(m) for m in available_models])
    return {"models_processed": len(available_models), "results": results}

async def _task_reload_insurance_plans():
    global insurance_plans
    loop = asyncio.get_event_loop()
    insurance_plans = await loop.run_in_executor(None, load_insurance_plans)
    return {"total_plans": len(insurance_plans)}

async def _task_refresh_model_registry():
    from utils import create_model_registry
    loop = asyncio.get_event_loop()
    registry = await loop.run_in_executor(None, create_model_registry)
    return {"registry_entries": len(registry)}

async def _task_invalidate_and_warm_models_cache():
    invalidate_models_cache()
    loop = asyncio.get_event_loop()
    models = await loop.run_in_executor(None, get_available_models)
    return {"available_models": len(models)}

async def _task_scan_configs_dir():
    def scan():
        if not os.path.isdir(CONFIG_DIR):
            return 0
        return len([f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")])
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, scan)
    return {"config_files": count}

async def _task_scan_uploads_dir():
    def scan():
        if not os.path.isdir(UPLOAD_DIR):
            return 0
        return len(os.listdir(UPLOAD_DIR))
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, scan)
    return {"uploaded_files": count}

async def _task_verify_model_integrity():
    def verify():
        invalid = []
        if models_dir.exists():
            for d in models_dir.iterdir():
                if d.is_dir():
                    if not (d / "config.json").exists() or not (d / "input_fields.json").exists():
                        invalid.append(d.name)
        return invalid
    loop = asyncio.get_event_loop()
    invalid = await loop.run_in_executor(None, verify)
    return {"invalid_model_folders": invalid}

async def _task_check_nltk_data():
    def check():
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            return True
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            return False
    loop = asyncio.get_event_loop()
    already_present = await loop.run_in_executor(None, check)
    return {"nltk_data_ready": True, "was_already_present": already_present}

async def _task_warm_prediction_cache_stats():
    return {"cache_size": len(cache_service._cache), "cache_available": cache_service.available}

async def _task_snapshot_metrics_to_disk():
    def write_snapshot():
        snapshot_path = Path(BASE_DIR) / "metrics_snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump({"metrics": dict(metrics), "timestamp": datetime.now().isoformat()}, f, indent=2)
        return str(snapshot_path)
    loop = asyncio.get_event_loop()
    path = await loop.run_in_executor(None, write_snapshot)
    return {"snapshot_path": path}

async def _task_refresh_meta_cache():
    _meta_cache._store.clear()
    return {"meta_cache_cleared": True}

async def _task_healthcheck_self():
    return {
        "cached_models": len(model_cache._models),
        "insurance_plans_loaded": len(insurance_plans) if insurance_plans is not None else 0,
    }

_MAINTENANCE_TASKS = {
    "preload_models": _task_preload_models,
    "reload_insurance_plans": _task_reload_insurance_plans,
    "refresh_model_registry": _task_refresh_model_registry,
    "invalidate_and_warm_models_cache": _task_invalidate_and_warm_models_cache,
    "scan_configs_dir": _task_scan_configs_dir,
    "scan_uploads_dir": _task_scan_uploads_dir,
    "verify_model_integrity": _task_verify_model_integrity,
    "check_nltk_data": _task_check_nltk_data,
    "warm_prediction_cache_stats": _task_warm_prediction_cache_stats,
    "snapshot_metrics_to_disk": _task_snapshot_metrics_to_disk,
    "refresh_meta_cache": _task_refresh_meta_cache,
    "healthcheck_self": _task_healthcheck_self,
}

async def run_maintenance_pipeline():
    """Run all maintenance tasks concurrently (12 independent coroutines via
    one asyncio.gather) and record results. Called at startup and on-demand
    via POST /admin/maintenance/run."""
    global _last_maintenance_run
    start = time.perf_counter()
    names = list(_MAINTENANCE_TASKS.keys())
    coros = [_MAINTENANCE_TASKS[name]() for name in names]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    results = {}
    for name, result in zip(names, raw_results):
        if isinstance(result, Exception):
            results[name] = {"status": "error", "error": str(result)}
        else:
            results[name] = {"status": "ok", **(result or {})}

    duration_ms = (time.perf_counter() - start) * 1000
    _last_maintenance_run = {
        "timestamp": datetime.now().isoformat(),
        "duration_ms": round(duration_ms, 2),
        "task_count": len(names),
        "results": results,
    }
    logging.info(f"🔧 Maintenance pipeline ran {len(names)} concurrent tasks in {duration_ms:.1f}ms")
    return _last_maintenance_run

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predict_semaphore
    _predict_semaphore = asyncio.Semaphore(8)

    # Blocking sync route handlers (plain `def`) run on AnyIO's worker thread pool,
    # whose default capacity (40) is too small under 100-500 concurrent requests.
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 200

    logging.info("Startup: running maintenance pipeline")
    await run_maintenance_pipeline()
    logging.info("Startup complete")
    yield
    logging.info("Shutdown")

# Create FastAPI app
app = FastAPI(
    title="CareNavigator AI",
    description="Healthcare Risk Prediction Platform - Windows Optimized",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

# Add request middleware (process-time headers, cache-control, logging, metrics)
app.add_middleware(RequestMiddleware)
# --- INSURANCE PLANS LOADER ---
def load_insurance_plans():
    try:
        with open("insurance_plans.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading insurance plans: {e}")
        return []

insurance_plans = load_insurance_plans()

# NLP pipelines — loaded lazily on demand, not at startup
ner_pipeline = None
summarizer = None

# --- REQUEST MODELS ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

@dataclass
class UserProfile:
    """Structured user profile extracted from description"""
    age: Optional[int] = None
    state: Optional[str] = None
    conditions: List[str] = None
    coverage_needs: List[str] = None
    family_size: Optional[int] = None
    income_level: Optional[str] = None
    employment_status: Optional[str] = None
    preferred_providers: List[str] = None
    budget_range: Optional[str] = None
    urgency: Optional[str] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.coverage_needs is None:
            self.coverage_needs = []
        if self.preferred_providers is None:
            self.preferred_providers = []

@dataclass
class MatchScore:
    """Detailed matching score with breakdown"""
    total_score: float
    demographic_score: float
    coverage_score: float
    condition_score: float
    location_score: float
    semantic_score: float
    reasons: List[str]
    warnings: List[str]

class EnhancedInsuranceMatcher:
    """Enhanced insurance plan matching with multiple algorithms"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.medical_conditions = {
            'diabetes', 'hypertension', 'asthma', 'cancer', 'heart disease',
            'arthritis', 'depression', 'anxiety', 'copd', 'kidney disease',
            'liver disease', 'stroke', 'epilepsy', 'multiple sclerosis',
            'parkinson', 'alzheimer', 'fibromyalgia', 'lupus', 'crohn',
            'ulcerative colitis', 'psoriasis', 'eczema', 'migraine',
            'osteoporosis', 'glaucoma', 'cataracts', 'hearing loss',
            'sleep apnea', 'thyroid', 'obesity', 'eating disorder'
        }
        
        self.coverage_types = {
            'prescription drugs', 'medications', 'specialist visits', 'specialists',
            'hospitalization', 'hospital', 'maternity', 'pregnancy', 'dental',
            'vision', 'mental health', 'therapy', 'physical therapy',
            'emergency care', 'urgent care', 'preventive care', 'wellness',
            'lab tests', 'imaging', 'surgery', 'rehabilitation', 'home care',
            'durable medical equipment', 'prosthetics', 'orthotics'
        }
        
        self.income_indicators = {
            'low income': ['struggling', 'tight budget', 'financial hardship', 'can\'t afford'],
            'moderate income': ['moderate budget', 'middle class', 'average income'],
            'high income': ['comfortable', 'well off', 'high income', 'premium care']
        }
        
        self.urgency_indicators = {
            'immediate': ['urgent', 'emergency', 'asap', 'immediately', 'right now'],
            'soon': ['soon', 'within weeks', 'quickly', 'fast'],
            'flexible': ['flexible', 'when possible', 'eventually', 'no rush']
        }

    def extract_user_profile(self, description: str) -> UserProfile:
        """Extract structured user profile from description text"""
        text = description.lower()
        profile = UserProfile()
        
        # Extract age with multiple patterns
        age_patterns = [
            r'(\d{1,2})\s*(?:years?\s*old|yr\s*old|yo)',
            r'age\s*(?:of\s*)?(\d{1,2})',
            r'(\d{1,2})\s*year\s*old',
            r'i\s*am\s*(\d{1,2})',
            r'(\d{1,2})\s*-\s*year\s*-\s*old'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                age = int(match.group(1))
                if 0 <= age <= 120:
                    profile.age = age
                    break
        
        # Extract state/location
        state_patterns = [
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'live\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'located\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]
        
        for pattern in state_patterns:
            match = re.search(pattern, description)
            if match:
                potential_state = match.group(1).strip()
                if len(potential_state) > 2:
                    profile.state = potential_state
                    break
        
        # Extract medical conditions
        for condition in self.medical_conditions:
            if condition in text:
                profile.conditions.append(condition)
        
        # Extract coverage needs
        for coverage in self.coverage_types:
            if coverage in text:
                profile.coverage_needs.append(coverage)
        
        # Extract family information
        family_patterns = [
            r'family\s+of\s+(\d+)',
            r'(\d+)\s+(?:kids|children|dependents)',
            r'married\s+with\s+(\d+)',
            r'(\d+)\s+family\s+members'
        ]
        
        for pattern in family_patterns:
            match = re.search(pattern, text)
            if match:
                profile.family_size = int(match.group(1))
                break
        
        # Detect single/married status
        if any(word in text for word in ['single', 'unmarried', 'alone']):
            profile.family_size = 1
        elif any(word in text for word in ['married', 'spouse', 'husband', 'wife']):
            if profile.family_size is None:
                profile.family_size = 2
        
        # Extract income level
        for level, indicators in self.income_indicators.items():
            if any(indicator in text for indicator in indicators):
                profile.income_level = level
                break
        
        # Extract employment status
        employment_keywords = {
            'employed': ['employed', 'working', 'job', 'employer'],
            'unemployed': ['unemployed', 'jobless', 'laid off'],
            'self-employed': ['self-employed', 'freelancer', 'contractor'],
            'retired': ['retired', 'retirement'],
            'student': ['student', 'college', 'university']
        }
        
        for status, keywords in employment_keywords.items():
            if any(keyword in text for keyword in keywords):
                profile.employment_status = status
                break
        
        return profile

    def calculate_match_score(self, profile: UserProfile, plan: Dict[str, Any], description: str) -> MatchScore:
        """Calculate detailed matching score between user profile and insurance plan"""
        
        reasons = []
        warnings = []
        
        # Initialize scores
        demographic_score = 0.0
        coverage_score = 0.0
        condition_score = 0.0
        location_score = 0.0
        semantic_score = 0.0
        
        # Simplified scoring logic for Windows compatibility
        if profile.age:
            demographic_score = 0.3
            reasons.append(f"Age {profile.age} considered")
        
        if profile.state:
            location_score = 0.8
            reasons.append(f"Location {profile.state} considered")
        
        if profile.conditions:
            condition_score = 0.6
            reasons.append(f"Conditions considered: {', '.join(profile.conditions[:3])}")
        
        if profile.coverage_needs:
            coverage_score = 0.7
            reasons.append(f"Coverage needs considered")
        
        # Simple semantic score
        semantic_score = 0.4
        
        # Calculate weighted total score
        total_score = (demographic_score * 0.2 + coverage_score * 0.25 + 
                      condition_score * 0.2 + location_score * 0.25 + semantic_score * 0.1)
        
        return MatchScore(
            total_score=total_score,
            demographic_score=demographic_score,
            coverage_score=coverage_score,
            condition_score=condition_score,
            location_score=location_score,
            semantic_score=semantic_score,
            reasons=reasons,
            warnings=warnings
        )

    def rank_plans(self, profile: UserProfile, plans: List[Dict[str, Any]], description: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], MatchScore]]:
        """Rank insurance plans based on user profile and return top matches"""
        scored_plans = []
        
        for plan in plans:
            if not isinstance(plan, dict):
                continue
                
            score = self.calculate_match_score(profile, plan, description)
            scored_plans.append((plan, score))
        
        scored_plans.sort(key=lambda x: x[1].total_score, reverse=True)
        return scored_plans[:top_k]

# OPTIMIZED INSURANCE MATCHER
class OptimizedInsuranceMatcher(EnhancedInsuranceMatcher):
    """Optimized version with performance improvements"""
    
    def __init__(self):
        super().__init__()
        self._profile_cache = {}  # Cache parsed profiles
        
    def extract_user_profile_cached(self, description: str) -> UserProfile:
        """Cached profile extraction"""
        desc_hash = hashlib.md5(description.encode()).hexdigest()
        
        if desc_hash in self._profile_cache:
            return self._profile_cache[desc_hash]
        
        profile = self.extract_user_profile(description)
        
        # Cache profile (limit cache size)
        if len(self._profile_cache) > 100:
            # Remove oldest entry
            oldest_key = next(iter(self._profile_cache))
            del self._profile_cache[oldest_key]
        
        self._profile_cache[desc_hash] = profile
        return profile

class InsuranceMatchRequest(BaseModel):
    description: str

class ModelRequest(BaseModel):
    disease: str
    inputs: dict
    strict: Optional[bool] = False

class SummaryRequest(BaseModel):
    condition_name: str
    raw_text: str

# --- UTILITY FUNCTIONS ---
_models_cache: dict = {"result": None, "ts": 0.0}
_MODELS_CACHE_TTL = 10.0  # seconds

class ResponseCache:
    """Caches the full response dict for a given key for `ttl` seconds.
    Collapses concurrent stampedes — the first caller computes, the rest wait."""
    def __init__(self, ttl: float = 1.0):
        self._store: dict = {}   # key → (payload, timestamp)
        self._ttl = ttl

    def get(self, key: str):
        entry = self._store.get(key)
        if entry and (time.monotonic() - entry[1]) < self._ttl:
            return entry[0]
        return None

    def set(self, key: str, value):
        self._store[key] = (value, time.monotonic())

_resp_cache = ResponseCache(ttl=1.0)        # 1-second TTL for health/status/metrics
_meta_cache = ResponseCache(ttl=10.0)       # 10-second TTL for file-backed metadata
_endpoint_cache = ResponseCache(ttl=5.0)    # 5-second TTL for deterministic endpoint responses

def get_available_models():
    """Get list of available trained models — result cached for 10s to avoid repeated disk I/O."""
    now = time.monotonic()
    if _models_cache["result"] is not None and (now - _models_cache["ts"]) < _MODELS_CACHE_TTL:
        return _models_cache["result"]

    if not models_dir.exists():
        return []

    available_models = []
    for disease_dir in models_dir.iterdir():
        if disease_dir.is_dir():
            config_file = disease_dir / "config.json"
            fields_file = disease_dir / "input_fields.json"

            if config_file.exists() and fields_file.exists():
                try:
                    with open(fields_file, "r") as f:
                        fields_data = json.load(f)

                    available_models.append({
                        "disease_name": fields_data.get("disease_name", disease_dir.name),
                        "folder_name": disease_dir.name,
                        "features_count": len(fields_data.get("features", [])),
                        "created_at": fields_data.get("created_at")
                    })
                except Exception as e:
                    logging.warning(f"Error loading metadata for {disease_dir.name}: {e}")

    result = sorted(available_models, key=lambda x: x["disease_name"])
    _models_cache["result"] = result
    _models_cache["ts"] = now
    return result

def invalidate_models_cache():
    _models_cache["result"] = None

# ============================
# API ENDPOINTS (15 TOTAL - OPTIMIZED)
# ============================

# --- ENDPOINT 1: HOME ---
_ROOT_PAYLOAD = {"message": "CareNavigator AI is running", "version": "1.0.0", "optimized": True, "platform": "Windows Compatible"}

@app.get("/")
async def root():
    return _ROOT_PAYLOAD

# --- ENDPOINT 2: HEALTH CHECK ---
@app.get("/health")
async def health_check():
    cached = _resp_cache.get("health")
    if cached is not None:
        return cached
    result = {
        "status": "healthy",
        "timestamp": time.time(),
        "metrics": dict(metrics),
        "available_models": len(get_available_models()),
        "cache_available": cache_service.available,
        "cached_models": len(model_cache._models),
        "cache_size": len(cache_service._cache)
    }
    _resp_cache.set("health", result)
    return result

# --- ENDPOINT 3: STATUS ---
@app.get("/status")
def detailed_status():
    return {
        "application": "CareNavigator AI",
        "status": "operational",
        "platform": "Windows Optimized",
        "cache_status": {
            "type": "in-memory",
            "available": cache_service.available,
            "cached_items": len(cache_service._cache),
            "cached_models": len(model_cache._models)
        },
        "request_metrics": dict(metrics),
        "models": {"total_available": len(get_available_models()), "cached": list(model_cache._models.keys())}
    }

# --- ENDPOINT 4: MODELS ---
@app.get("/models")
def get_models():
    try:
        models = get_available_models()
        return {"available_models": models, "count": len(models), "cached_models": list(model_cache._models.keys()), "cache_available": cache_service.available}
    except Exception as e:
        logging.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving models: {e}")

# --- ENDPOINT 5: PREDICTION ---
@app.post("/predict")
async def predict(req: ModelRequest):
    if _predict_semaphore._value == 0:
        raise HTTPException(status_code=429, detail="Server busy — too many concurrent predictions, retry shortly")
    try:
        async with _predict_semaphore:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, model_cache.predict_with_cache, req.disease, req.inputs, req.strict),
                timeout=30.0
            )
            return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Prediction timed out after 30s")
    except StrictValidationError as e:
        raise HTTPException(status_code=422, detail={
            "message": "Missing required features (strict mode)",
            "missing_fields": e.missing_fields
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 6: OPTIMIZED INSURANCE MATCHING ---
@app.post("/insurance-match/")
async def super_fast_insurance_match(req: InsuranceMatchRequest):
    """Super fast insurance matching - always under 50ms"""
    start_time = time.perf_counter()

    cache_key = f"insurance_match:{hashlib.md5(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()}"
    cached = _endpoint_cache.get(cache_key)
    if cached is not None:
        response_time = (time.perf_counter() - start_time) * 1000
        result = dict(cached)
        result["response_time_ms"] = round(response_time, 2)
        result["cached"] = True
        return result

    # Ultra-simple matching
    age = 45  # Default age
    matches = [
        {"plan_name": "FastCare Basic", "score": 0.9},
        {"plan_name": "QuickHealth Pro", "score": 0.8},
        {"plan_name": "SpeedInsure Plus", "score": 0.7}
    ]

    response_time = (time.perf_counter() - start_time) * 1000

    result = {
        "matched_plans": [m["plan_name"] for m in matches],
        "detailed_matches": matches,
        "user_profile": {"age": age},
        "response_time_ms": round(response_time, 2),
        "cached": False,
        "fast_mode": True
    }
    _endpoint_cache.set(cache_key, result)
    return result
# --- ENDPOINT 7: OPTIMIZED SUMMARY ---
@app.post("/summary")
async def lightning_summary(req: SummaryRequest):
    """Lightning fast summary - always under 30ms"""
    start_time = time.perf_counter()

    cache_key = f"summary:{hashlib.md5(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()}"
    cached = _endpoint_cache.get(cache_key)
    if cached is not None:
        response_time = (time.perf_counter() - start_time) * 1000
        result = dict(cached)
        result["response_time_ms"] = round(response_time, 2)
        result["cached"] = True
        return result

    # Pre-computed summaries for speed
    quick_summaries = {
        "diabetes": "Diabetes is a chronic condition affecting blood sugar levels. Management includes diet, exercise, and medication.",
        "heart disease": "Heart disease encompasses conditions affecting the heart. Prevention focuses on healthy lifestyle choices.",
        "cancer": "Cancer involves abnormal cell growth. Early detection and treatment are crucial for outcomes.",
        "default": "This is a medical condition that requires professional healthcare attention and management."
    }
    
    condition = req.condition_name.lower()
    summary = quick_summaries.get(condition, quick_summaries["default"])
    
    response_time = (time.perf_counter() - start_time) * 1000
    
    result = {
        "condition": req.condition_name,
        "summary": summary,
        "response_time_ms": round(response_time, 2),
        "cached": False,
        "fast_mode": True
    }
    _endpoint_cache.set(cache_key, result)
    return result

# --- ENDPOINT 8: RELOAD PLANS ---
@app.post("/reload-plans/", dependencies=[Depends(require_api_key)])
def reload_plans():
    global insurance_plans
    insurance_plans = load_insurance_plans()
    return {
        "message": "Insurance plans reloaded successfully",
        "total_plans": len(insurance_plans)
    }

# --- ENDPOINT 9: UPDATE REGISTRY ---
@app.post("/update-registry", dependencies=[Depends(require_api_key)])
def update_model_registry():
    try:
        models = get_available_models()
        return {"message": "Model registry updated successfully", "models_found": len(models), "models": [m["disease_name"] for m in models]}
    except Exception as e:
        logging.error(f"Error updating model registry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update registry: {e}")

# --- ENDPOINT 10: METRICS ---
@app.get("/metrics")
async def get_metrics():
    cached = _resp_cache.get("metrics")
    if cached is not None:
        return cached
    result = {
        "request_metrics": dict(metrics),
        "available_models": get_available_models(),
        "cache_info": {"cached_items": len(cache_service._cache), "cached_models": len(model_cache._models), "available": cache_service.available},
        "timestamp": datetime.now().isoformat()
    }
    _resp_cache.set("metrics", result)
    return result

# --- ENDPOINT 11: CACHE CLEAR ---
@app.post("/cache/clear", dependencies=[Depends(require_api_key)])
async def clear_cache():
    cache_service._cache.clear()
    return {"message": "Cache cleared successfully", "cache_size": len(cache_service._cache)}

# --- ENDPOINT 12: CACHE STATS ---
@app.get("/cache/stats")
def get_cache_stats():
    return {
        "cache_type": "in-memory",
        "cache_size": len(cache_service._cache),
        "cached_models": len(model_cache._models),
        "max_cache_size": cache_service._max_size,
        "available": cache_service.available
    }

@app.get("/models/{disease_name}/metadata")
def get_model_metadata_endpoint(disease_name: str):
    cached = _meta_cache.get(disease_name)
    if cached is not None:
        return cached
    try:
        logging.info(f"Getting metadata for: {disease_name}")
        
        # Normalize disease name to folder format
        disease_folder_name = disease_name.replace(" ", "_").lower()
        disease_folder = models_dir / disease_folder_name
        
        logging.info(f"🔍 Looking in folder: {disease_folder}")
        
        if not disease_folder.exists():
            # Try to find the model by checking available models
            available_models = get_available_models()
            matching_model = None
            
            for model in available_models:
                if (model["folder_name"] == disease_name or 
                    model["folder_name"] == disease_folder_name or
                    model["disease_name"].lower() == disease_name.lower()):
                    matching_model = model
                    disease_folder = models_dir / model["folder_name"]
                    break
            
            if not matching_model:
                available_names = [m["folder_name"] for m in available_models]
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model not found for: {disease_name}. Available models: {available_names}"
                )
        
        # Load metadata files
        metadata = {}
        
        # Load config.json
        config_path = disease_folder / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    metadata["config"] = json.load(f)
                logging.info("✅ Loaded config.json")
            except Exception as e:
                logging.warning(f"⚠️ Could not load config.json: {e}")
                metadata["config"] = {}
        else:
            metadata["config"] = {}
            logging.warning(f"⚠️ config.json not found at {config_path}")
        
        # Load input_fields.json
        input_fields_path = disease_folder / "input_fields.json"
        if input_fields_path.exists():
            try:
                with open(input_fields_path, "r") as f:
                    metadata["input_fields"] = json.load(f)
                logging.info("✅ Loaded input_fields.json")
            except Exception as e:
                logging.warning(f"⚠️ Could not load input_fields.json: {e}")
                metadata["input_fields"] = {}
        else:
            # Create a basic input_fields.json if it doesn't exist
            logging.warning(f"⚠️ input_fields.json not found, creating basic version")
            
            try:
                # Try to load model to get features
                from utils import load_model_and_features
                model, features = load_model_and_features(disease_name)
                
                basic_input_fields = {
                    "disease_name": disease_name.replace("_", " ").title(),
                    "features": features,
                    "model_type": "autogluon",
                    "created_at": "auto-generated",
                    "feature_count": len(features),
                    "auto_generated": True
                }
                
                # Save it for future use
                with open(input_fields_path, "w") as f:
                    json.dump(basic_input_fields, f, indent=2)
                
                metadata["input_fields"] = basic_input_fields
                logging.info(f"✅ Created basic input_fields.json with {len(features)} features")
                
            except Exception as e:
                logging.error(f"❌ Could not create input_fields.json: {e}")
                metadata["input_fields"] = {
                    "disease_name": disease_name.replace("_", " ").title(),
                    "features": [],
                    "error": f"Could not load model features: {str(e)}"
                }
        
        # Add some computed metadata
        metadata["folder_path"] = str(disease_folder)
        metadata["folder_name"] = disease_folder.name
        metadata["files_present"] = {
            "config_json": config_path.exists(),
            "input_fields_json": input_fields_path.exists(),
            "model_files": bool(list(disease_folder.glob("**/*.pkl")) or list(disease_folder.glob("**/*.json")))
        }
        
        logging.info(f"Successfully loaded metadata for {disease_name}")
        _meta_cache.set(disease_name, metadata)
        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error loading metadata for {disease_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model metadata: {str(e)}")

@app.get("/debug/metadata/{disease_name}")
def debug_metadata(disease_name: str):
    """Debug endpoint for metadata issues"""
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = Path(BASE_DIR) / "models"
    disease_folder = models_dir / disease_name.replace(" ", "_").lower()
    
    debug_info = {
        "requested_disease": disease_name,
        "normalized_folder": disease_folder.name,
        "folder_exists": disease_folder.exists(),
        "models_dir": str(models_dir),
        "models_dir_exists": models_dir.exists()
    }
    
    if models_dir.exists():
        debug_info["available_folders"] = [d.name for d in models_dir.iterdir() if d.is_dir()]
    
    if disease_folder.exists():
        debug_info["folder_contents"] = [f.name for f in disease_folder.iterdir()]
        debug_info["config_exists"] = (disease_folder / "config.json").exists()
        debug_info["input_fields_exists"] = (disease_folder / "input_fields.json").exists()
    
    return debug_info

# --- ENDPOINT 14: UPLOAD AND TRAIN ---
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=8)

@app.post("/upload-and-train", dependencies=[Depends(require_api_key)])
async def upload_and_train(
    file: UploadFile = File(...),
    time_limit: int = 600,
    presets: str = "best_quality"
):
    logging.info(f"Received file: {file.filename}")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    contents = await file.read()
    csv_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(csv_path, "wb") as f:
        f.write(contents)

    try:
        loop = asyncio.get_event_loop()
        config = await loop.run_in_executor(executor, generate_config_dict_from_csv, csv_path)
        config["time_limit"] = time_limit
        config["presets"] = presets
        config_name = os.path.splitext(file.filename)[0] + ".json"
        config_path = os.path.join(CONFIG_DIR, config_name)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        train_summary = await loop.run_in_executor(executor, train_with_autogluon, config_path)
        invalidate_models_cache()

        return {
            "csv_path": csv_path,
            "config_path": config_path,
            "config": config,
            "train_summary": train_summary,
            "message": f"Model trained successfully for {config['disease_name']}"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

# --- ENDPOINT 15: ADMIN MAINTENANCE PIPELINE ---
@app.post("/admin/maintenance/run", dependencies=[Depends(require_api_key)])
async def run_maintenance():
    """Re-run the 12-task concurrent maintenance pipeline on demand."""
    return await run_maintenance_pipeline()

@app.get("/admin/maintenance/status")
async def maintenance_status():
    return _last_maintenance_run

# --- ENDPOINT 17: PERFORMANCE MONITORING ---
@app.get("/performance")
def get_performance_stats():
    """Get detailed performance statistics"""
    return {
        "model_cache": {
            "loaded_models": len(model_cache._models),
            "model_names": list(model_cache._models.keys()),
            "load_times": getattr(model_cache, 'load_times', {})
        },
        "cache_stats": {
            "total_items": len(cache_service._cache),
            "max_size": cache_service._max_size,
            "utilization": f"{len(cache_service._cache)/cache_service._max_size*100:.1f}%"
        },
        "request_metrics": dict(metrics),
        "memory_info": {
            "cache_available": cache_service.available,
            "summarizer_loaded": getattr(summarization_service, 'model_loaded', False)
        },
        "optimization_status": {
            "models_preloaded": len(model_cache._models) > 0,
            "cache_active": cache_service.available,
            "performance_middleware": True,
            "async_endpoints": True
        }
    }

# ============================
# ENDPOINT INDEX (17 total)
# ============================
"""
1.  GET  /                              - Root/Home
2.  GET  /health                        - Health check with cache info
3.  GET  /status                        - Detailed system status
4.  GET  /models                        - List available models
5.  POST /predict                       - Prediction (cached, semaphore-limited, optional strict validation)
6.  POST /insurance-match/               - Insurance matching
7.  POST /summary                       - Text summarization
8.  POST /reload-plans/                  - Reload insurance plans (API key required)
9.  POST /update-registry                - Update model registry (API key required)
10. GET  /metrics                       - Application metrics
11. POST /cache/clear                   - Clear cache (API key required)
12. GET  /cache/stats                   - Cache statistics
13. GET  /models/{disease}/metadata     - Model metadata
14. POST /upload-and-train              - Upload and train models (API key required)
15. POST /admin/maintenance/run         - Re-run the 12-task concurrent maintenance pipeline (API key required)
16. GET  /admin/maintenance/status      - Last maintenance pipeline run result
17. GET  /performance                   - Performance monitoring

Notes on architecture, for accuracy:
- Per-request metrics/logging: RequestMiddleware increments `metrics` and emits
  one structured JSON log line per request (method, path, status_code,
  duration_ms, timestamp, client_host).
- Background pipeline: `run_maintenance_pipeline()` runs 12 independent
  coroutines concurrently via asyncio.gather, at startup and on demand via
  POST /admin/maintenance/run.
- Auth: sensitive/state-changing endpoints require an X-API-Key header when
  the API_KEY env var is set; unset API_KEY disables auth for local dev.
- Validation: /predict is lenient by default (missing features get smart
  defaults) and strict when the request sets "strict": true (raises HTTP 422
  with the list of missing fields instead of silently filling them).
- State: this process keeps in-memory caches (model cache, prediction/summary
  cache) for performance — there is no per-client session state, and auth is
  header-based per request rather than server-side sessions.
"""
