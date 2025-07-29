import asyncio
from functools import lru_cache
from typing import Dict, Any
import joblib
import torch
import os

class ModelCache:
    def __init__(self):
        self._models = {}
        self._lock = asyncio.Lock()
    
    async def load_model(self, model_name: str):
        async with self._lock:
            if model_name not in self._models:
                # Adjust path to match your models directory
                model_path = f"models/{model_name}.pkl"
                if os.path.exists(model_path):
                    self._models[model_name] = joblib.load(model_path)
                else:
                    # Fallback to your existing model loading logic
                    from backend.model_training.utils.model_utils import load_model_by_disease
                    self._models[model_name] = load_model_by_disease(model_name)
            return self._models[model_name]
    
    def get_model_sync(self, model_name: str):
        return self._models.get(model_name)

# Global cache instance
model_cache = ModelCache()