from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse
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
from transformers import pipeline
from dotenv import load_dotenv
from utils import load_model_and_features
from auto_config_generator import generate_config_dict_from_csv
import tempfile
from train_model import train_with_autogluon
import traceback
from typing import List, Dict, Any, Optional,Tuple
from dataclasses import dataclass
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = Path(BASE_DIR) / "models"         # new (Path object)
# CareNavigator API - FastAPI Application

# --- ENV & LOGGING SETUP ---
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- INSURANCE PLANS LOADER ---
def load_insurance_plans():
    try:
        with open("insurance_plans.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading insurance plans: {e}")
        return []

insurance_plans = load_insurance_plans()

# --- NLP Pipelines ---
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
summarizer = pipeline("summarization", model="t5-small")

# --- REQUEST MODELS ---
# Download required NLTK data (run once)
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
                if 0 <= age <= 120:  # Reasonable age range
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
            match = re.search(pattern, description)  # Case sensitive for state names
            if match:
                potential_state = match.group(1).strip()
                if len(potential_state) > 2:  # Avoid abbreviations for now
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
        
        # Extract budget information
        budget_patterns = [
            r'budget\s+(?:of\s+)?[\$]?(\d+)',
            r'afford\s+[\$]?(\d+)',
            r'spend\s+[\$]?(\d+)',
            r'[\$](\d+)\s+(?:per\s+month|monthly)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, text)
            if match:
                amount = int(match.group(1))
                if amount < 200:
                    profile.budget_range = 'low'
                elif amount < 500:
                    profile.budget_range = 'moderate'
                else:
                    profile.budget_range = 'high'
                break
        
        # Extract urgency
        for urgency, indicators in self.urgency_indicators.items():
            if any(indicator in text for indicator in indicators):
                profile.urgency = urgency
                break
        
        # Extract preferred providers
        provider_patterns = [
            r'prefer\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'like\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'want\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:provider|doctor|hospital)'
        ]
        
        for pattern in provider_patterns:
            matches = re.findall(pattern, description)
            profile.preferred_providers.extend(matches)
        
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
        
        # Demographic scoring (age, family size, income)
        if profile.age:
            # Check if plan has age restrictions
            min_age = plan.get('min_age')
            max_age = plan.get('max_age')
            
            if min_age is not None and max_age is not None:
                # Both age limits exist
                if min_age <= profile.age <= max_age:
                    demographic_score += 0.3
                    reasons.append(f"Age {profile.age} fits plan requirements ({min_age}-{max_age})")
                else:
                    warnings.append(f"Age {profile.age} outside plan range ({min_age}-{max_age})")
            elif min_age is not None:
                # Only minimum age exists
                if profile.age >= min_age:
                    demographic_score += 0.2
                    reasons.append(f"Meets minimum age requirement ({min_age}+)")
                else:
                    warnings.append(f"Below minimum age requirement ({min_age})")
            elif max_age is not None:
                # Only maximum age exists
                if profile.age <= max_age:
                    demographic_score += 0.2
                    reasons.append(f"Within maximum age limit ({max_age})")
                else:
                    warnings.append(f"Above maximum age limit ({max_age})")
            else:
                # No age restrictions
                demographic_score += 0.1
                reasons.append("No age restrictions")
        
        # Family size matching
        if profile.family_size:
            family_plans_available = plan.get('family_plans', True)  # Default to True if not specified
            
            if profile.family_size > 1 and family_plans_available:
                demographic_score += 0.2
                reasons.append("Family plan available")
            elif profile.family_size == 1:
                demographic_score += 0.1
                reasons.append("Individual plan suitable")
            elif profile.family_size > 1 and not family_plans_available:
                warnings.append("Family plans may not be available")
        
        # Location scoring
        if profile.state:
            plan_states = plan.get('states', [])
            coverage_area = plan.get('coverage_area', '')
            
            # Check if plan covers the user's state
            state_covered = False
            
            if plan_states:
                # Check exact state matches
                state_covered = any(
                    profile.state.lower() in state.lower() or state.lower() in profile.state.lower()
                    for state in plan_states
                )
            elif coverage_area:
                # Check coverage area descriptions
                coverage_keywords = ['nationwide', 'national', 'all states', profile.state.lower()]
                state_covered = any(keyword in coverage_area.lower() for keyword in coverage_keywords)
            
            if state_covered:
                location_score = 1.0
                reasons.append(f"Available in {profile.state}")
            elif not plan_states and not coverage_area:
                # No location restrictions specified
                location_score = 0.7
                reasons.append("No location restrictions specified")
            else:
                location_score = 0.0
                warnings.append(f"May not be available in {profile.state}")
        else:
            # No state specified by user
            location_score = 0.5
        
        # Coverage matching
        if profile.coverage_needs:
            plan_coverage = plan.get('coverage', [])
            
            if plan_coverage:
                matched_coverage = []
                for need in profile.coverage_needs:
                    for plan_cov in plan_coverage:
                        if (need.lower() in plan_cov.lower() or 
                            plan_cov.lower() in need.lower() or
                            any(word in plan_cov.lower() for word in need.lower().split())):
                            matched_coverage.append(need)
                            break
                
                if matched_coverage:
                    coverage_score = len(matched_coverage) / len(profile.coverage_needs)
                    reasons.append(f"Covers: {', '.join(matched_coverage)}")
                
                # Check for missing coverage
                missing_coverage = [need for need in profile.coverage_needs if need not in matched_coverage]
                if missing_coverage:
                    warnings.append(f"May not cover: {', '.join(missing_coverage)}")
            else:
                # No coverage details available
                coverage_score = 0.3
                reasons.append("Coverage details not specified")
        
        # Condition-specific matching
        if profile.conditions:
            plan_conditions = plan.get('conditions', [])
            
            if plan_conditions:
                matched_conditions = []
                for condition in profile.conditions:
                    for plan_condition in plan_conditions:
                        if (condition.lower() in plan_condition.lower() or 
                            plan_condition.lower() in condition.lower()):
                            matched_conditions.append(condition)
                            break
                
                if matched_conditions:
                    condition_score = len(matched_conditions) / len(profile.conditions)
                    reasons.append(f"Specialized for: {', '.join(matched_conditions)}")
                else:
                    warnings.append("No specific coverage mentioned for your conditions")
            else:
                # No specific conditions mentioned
                condition_score = 0.2
        
        # Budget matching (if available in plan data)
        budget_score = 0.0
        if profile.budget_range and 'monthly_premium' in plan:
            try:
                monthly_premium = float(plan['monthly_premium'])
                
                if profile.budget_range == 'low' and monthly_premium < 200:
                    budget_score = 0.3
                    reasons.append("Fits low budget range")
                elif profile.budget_range == 'moderate' and 200 <= monthly_premium < 400:
                    budget_score = 0.3
                    reasons.append("Fits moderate budget range")
                elif profile.budget_range == 'high' and monthly_premium >= 300:
                    budget_score = 0.3
                    reasons.append("Premium plan within budget")
                elif monthly_premium > 500:
                    warnings.append("Higher premium plan")
            except (ValueError, TypeError):
                pass
        
        # Semantic similarity using TF-IDF
        try:
            plan_text = f"{plan.get('name', '')} {plan.get('description', '')} {' '.join(plan.get('coverage', []))}"
            
            if len(plan_text.strip()) > 0 and len(description.strip()) > 0:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
                vectors = vectorizer.fit_transform([description, plan_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                semantic_score = similarity
                
                if similarity > 0.3:
                    reasons.append(f"High text similarity ({similarity:.2f})")
            else:
                semantic_score = 0.1
        except Exception as e:
            logging.warning(f"Semantic similarity calculation failed: {e}")
            semantic_score = 0.1
        
        # Calculate weighted total score
        weights = {
            'demographic': 0.15,
            'coverage': 0.25,
            'condition': 0.20,
            'location': 0.25,
            'semantic': 0.10,
            'budget': 0.05
        }
        
        total_score = (
            demographic_score * weights['demographic'] +
            coverage_score * weights['coverage'] +
            condition_score * weights['condition'] +
            location_score * weights['location'] +
            semantic_score * weights['semantic'] +
            budget_score * weights['budget']
        )
        
        # Ensure score is between 0 and 1
        total_score = max(0.0, min(1.0, total_score))
        
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
        
        # Sort by total score (descending)
        scored_plans.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return scored_plans[:top_k]

class InsuranceMatchRequest(BaseModel):
    description: str

class ModelRequest(BaseModel):
    disease: str
    inputs: dict

class ExplainRequest(BaseModel):
    inputs: dict

class SummaryRequest(BaseModel):
    condition_name: str
    raw_text: str

# --- UTILITY FUNCTIONS FOR MODEL METADATA ---
def save_model_metadata(disease_name: str, config: dict, features: List[str], feature_types: Dict[str, str] = None):
    """Save model metadata including input fields and their types to the disease-specific folder"""
    disease_folder = models_dir / disease_name.replace(" ", "_").lower()
    disease_folder.mkdir(parents=True, exist_ok=True)
    
    # Save input fields metadata
    input_fields_data = {
        "disease_name": disease_name,
        "features": features,
        "feature_types": feature_types or {},
        "target_column": config.get("target_column"),
        "categorical_columns": config.get("categorical_columns", []),
        "numerical_columns": config.get("numerical_columns", []),
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    input_fields_path = disease_folder / "input_fields.json"
    with open(input_fields_path, "w") as f:
        json.dump(input_fields_data, f, indent=4)
    
    # Save full config
    config_path = disease_folder / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    logging.info(f"Saved metadata for {disease_name} in {disease_folder}")
    return input_fields_path, config_path

def load_model_metadata(disease_name: str):
    """Load model metadata for a specific disease"""
    disease_folder = models_dir / disease_name.replace(" ", "_").lower()
    
    input_fields_path = disease_folder / "input_fields.json"
    config_path = disease_folder / "config.json"
    
    metadata = {}
    
    if input_fields_path.exists():
        with open(input_fields_path, "r") as f:
            metadata["input_fields"] = json.load(f)
    
    if config_path.exists():
        with open(config_path, "r") as f:
            metadata["config"] = json.load(f)
    
    return metadata

def get_available_models():
    """Get list of available trained models"""
    print("DEBUG: Checking available models in directory:", models_dir)
    if not models_dir.exists():
        return []
    
    available_models = []
    for disease_dir in models_dir.iterdir():
        if disease_dir.is_dir():
            # Check if the disease folder has required files
            config_file = disease_dir / "config.json"
            fields_file = disease_dir / "input_fields.json"
            model_file = disease_dir / "predictor.pkl"
            
            if config_file.exists() and fields_file.exists():
                # Load basic info
                try:
                    with open(fields_file, "r") as f:
                        fields_data = json.load(f)
                    
                    available_models.append({
                        "disease_name": fields_data.get("disease_name", disease_dir.name),
                        "folder_name": disease_dir.name,
                        "features_count": len(fields_data.get("features", [])),
                        "created_at": fields_data.get("created_at"),
                        "has_model": model_file.exists() or any(disease_dir.glob("*.pkl"))
                    })
                except Exception as e:
                    logging.warning(f"Error loading metadata for {disease_dir.name}: {e}")
    
    return sorted(available_models, key=lambda x: x["disease_name"])

def infer_feature_types_from_sample(df: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]):
    """Infer feature types from sample data for better input field rendering"""
    feature_types = {}
    
    for col in df.columns:
        if col in categorical_cols:
            # Get unique values for categorical fields (limit to reasonable number)
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 20:  # Only store if reasonable number of categories
                feature_types[col] = {
                    "type": "categorical",
                    "options": sorted([str(v) for v in unique_vals])
                }
            else:
                feature_types[col] = {"type": "categorical", "options": []}
        elif col in numerical_cols:
            feature_types[col] = {
                "type": "numerical",
                "min": float(df[col].min()) if not df[col].isna().all() else 0,
                "max": float(df[col].max()) if not df[col].isna().all() else 100,
                "mean": float(df[col].mean()) if not df[col].isna().all() else 50
            }
        else:
            # Try to infer
            if df[col].dtype in ['object', 'category']:
                feature_types[col] = {"type": "categorical", "options": []}
            else:
                feature_types[col] = {"type": "numerical", "min": 0, "max": 100, "mean": 50}
    
    return feature_types

# --- HOME ---
@app.get("/")
async def root():
    return {"message": "CareNavigator API is running"}

# --- NEW ENDPOINT: GET AVAILABLE MODELS ---
@app.get("/models")
async def get_models():
    """Get list of all available trained models"""
    try:
        models = get_available_models()
        return {
            "available_models": models,
            "count": len(models)
        }
    except Exception as e:
        logging.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving models: {e}")

# --- NEW ENDPOINT: GET MODEL METADATA ---
@app.get("/models/{disease_name}/metadata")
async def get_model_metadata(disease_name: str):
    """Get metadata for a specific disease model including input fields"""
    try:
        metadata = load_model_metadata(disease_name)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Model metadata not found for disease: {disease_name}")
        
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error loading metadata for {disease_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model metadata: {e}")

# --- INSURANCE MATCH ENDPOINT ---
# Enhanced FastAPI endpoint
@app.post("/insurance-match/")
def enhanced_insurance_match(req: InsuranceMatchRequest):
    """Enhanced insurance matching with detailed scoring and explanations"""
    
    try:
        matcher = EnhancedInsuranceMatcher()
        
        # Extract structured user profile
        profile = matcher.extract_user_profile(req.description)
        
        # Get ranked matches
        ranked_matches = matcher.rank_plans(profile, insurance_plans, req.description, top_k=5)
        
        if not ranked_matches:
            return {
                "matched_plans": [],
                "user_profile": profile.__dict__,
                "explanation": "No suitable plans found based on your requirements.",
                "suggestions": [
                    "Try expanding your location preferences",
                    "Consider plans with different coverage options",
                    "Check if you qualify for government assistance programs"
                ]
            }
        
        # Prepare response
        matches = []
        for plan, score in ranked_matches:
            matches.append({
                "plan_name": plan.get("name", "Unknown Plan"),
                "description": plan.get("description", ""),
                "score": round(score.total_score, 3),
                "score_breakdown": {
                    "demographic": round(score.demographic_score, 3),
                    "coverage": round(score.coverage_score, 3),
                    "conditions": round(score.condition_score, 3),
                    "location": round(score.location_score, 3),
                    "semantic": round(score.semantic_score, 3)
                },
                "reasons": score.reasons,
                "warnings": score.warnings,
                "plan_details": {
                    "states": plan.get("states", []),
                    "coverage": plan.get("coverage", []),
                    "conditions": plan.get("conditions", []),
                    "min_age": plan.get("min_age"),
                    "max_age": plan.get("max_age")
                }
            })
        
        # Generate explanation
        best_match = ranked_matches[0]
        explanation_parts = []
        
        if best_match[1].reasons:
            explanation_parts.append(f"Best match: {best_match[0].get('name', 'Top Plan')} because it " + 
                                   ", ".join(best_match[1].reasons).lower())
        
        if len(ranked_matches) > 1:
            explanation_parts.append(f"Found {len(ranked_matches)} suitable options ranked by compatibility.")
        
        explanation = " ".join(explanation_parts) if explanation_parts else "Plans ranked by compatibility with your profile."
        
        return {
            "matched_plans": [match["plan_name"] for match in matches],
            "detailed_matches": matches,
            "user_profile": {
                "age": profile.age,
                "state": profile.state,
                "conditions": profile.conditions,
                "coverage_needs": profile.coverage_needs,
                "family_size": profile.family_size,
                "income_level": profile.income_level,
                "employment_status": profile.employment_status,
                "budget_range": profile.budget_range,
                "urgency": profile.urgency
            },
            "explanation": explanation,
            "matching_algorithm": "Enhanced multi-factor scoring with semantic analysis",
            "total_plans_evaluated": len(insurance_plans)
        }
        
    except Exception as e:
        logging.error(f"Enhanced insurance matching error: {e}")
        raise HTTPException(status_code=500, detail=f"Insurance matching failed: {e}")

# --- RELOAD INSURANCE PLANS ENDPOINT ---
@app.post("/reload-plans/")
def reload_plans():
    global insurance_plans
    insurance_plans = load_insurance_plans()
    return {"message": "Insurance plans reloaded."}

# --- ML ENDPOINTS (predict/explain/summary) ---
"""
@app.post("/predict")
def predict(req: ModelRequest):
    try:
        print(f"DEBUG: Received prediction request for disease: {req.disease}")
        model, expected_inputs = load_model_and_features(req.disease)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Fill missing with defaults as needed for inference
    row = {}
    for col in expected_inputs:
        value = req.inputs.get(col, None)
        if value is None:
            if col == "Unnamed: 0":
                row[col] = 0
            elif col.startswith("Hospital") or col.startswith("County") or col.startswith("location"):
                row[col] = ""
            else:
                row[col] = 0
        else:
            # Convert string inputs to appropriate types
            try:
                # Try to convert to float for numerical fields
                if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    row[col] = float(value)
                else:
                    row[col] = value
            except:
                row[col] = value
    
    df = pd.DataFrame([row])
    print("DEBUG: PREDICT DataFrame columns:", df.columns.tolist())
    prediction = model.predict(df)
    
    # Get prediction probabilities if available
    try:
        probabilities = model.predict_proba(df)
        prob_dict = {f"class_{i}": float(prob) for i, prob in enumerate(probabilities.iloc[0])}
    except:
        prob_dict = None
    
    return {
        "prediction": prediction.tolist()[0] if hasattr(prediction, 'tolist') else str(prediction.iloc[0]),
        "probabilities": prob_dict,
        "model": req.disease,
        "input_used": list(df.columns),
        "input_values": row
    }
"""
@app.post("/predict")
def predict(req: ModelRequest):
    try:
        print(f"DEBUG: Received prediction request for disease: {req.disease}")
        
        # Use the improved validation function
        from utils import validate_prediction_inputs, load_model_and_features
        
        # Validate and prepare inputs
        row, missing_features, expected_inputs = validate_prediction_inputs(req.disease, req.inputs)
        
        # Load model
        model, _ = load_model_and_features(req.disease)
        
        # Create DataFrame for prediction
        df = pd.DataFrame([row])
        print(f"DEBUG: PREDICT DataFrame columns: {df.columns.tolist()}")
        print(f"DEBUG: Input shape: {df.shape}")
        if missing_features:
            print(f"DEBUG: Missing features filled with defaults: {missing_features}")
        
        # Make prediction
        prediction = model.predict(df)
        
        # Get prediction probabilities if available
        probabilities = None
        try:
            probabilities = model.predict_proba(df)
            if probabilities is not None:
                prob_dict = {f"class_{i}": float(prob) for i, prob in enumerate(probabilities.iloc[0])}
            else:
                prob_dict = None
        except Exception as e:
            print(f"DEBUG: Could not get probabilities: {e}")
            prob_dict = None
        
        # Format prediction result
        if hasattr(prediction, 'tolist'):
            pred_result = prediction.tolist()[0]
        elif hasattr(prediction, 'iloc'):
            pred_result = str(prediction.iloc[0])
        else:
            pred_result = str(prediction)
        
        return {
            "prediction": pred_result,
            "probabilities": prob_dict,
            "model": req.disease,
            "input_features_used": list(df.columns),
            "input_values": row,
            "missing_features_filled": missing_features,
            "total_features": len(expected_inputs),
            "status": "success"
        }
        
    except ValueError as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"ERROR: Unexpected error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


"""
@app.post("/shap-explain")
def shap_explain(req: ExplainRequest):
]    try:
        # This is a placeholder - you'll need to adapt based on your specific model loading
        input_df = pd.DataFrame([req.inputs])
        
        # For now, return a mock response
        return {
            "message": "SHAP explanation not yet implemented for this model",
            "input_features": list(req.inputs.keys()),
            "explanation": "Feature importance analysis would appear here"
        }
    except Exception as e:
        logging.error(f"SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {e}")
"""

# --- UPLOAD AND TRAIN ENDPOINTS ---
UPLOAD_DIR = "uploads"
CONFIG_DIR = "configs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

"""
@app.post("/upload-and-train")
async def upload_and_train(file: UploadFile = File(...)):
    print(f"✅ Received file: {file.filename}")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    contents = await file.read()
    csv_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(csv_path, "wb") as f:
        f.write(contents)

    try:
        # Generate config dict and save JSON
        config = generate_config_dict_from_csv(csv_path)
        config_name = os.path.splitext(file.filename)[0] + ".json"
        config_path = os.path.join(CONFIG_DIR, config_name)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Train the model
        train_summary = train_with_autogluon(config_path)
        
        # Load the trained predictor
        disease_name = config["disease_name"].replace(" ", "_").lower()
        predictor_path = f'models/{disease_name}/'
        predictor = TabularPredictor.load(predictor_path)
        
        # Get features and infer types from the original data
        features = predictor.feature_metadata.get_features()
        
        # Load original data to infer feature types
        df = pd.read_csv(csv_path)
        feature_types = infer_feature_types_from_sample(
            df, 
            config.get("categorical_columns", []), 
            config.get("numerical_columns", [])
        )
        
        # Save model metadata including input fields
        input_fields_path, config_path_saved = save_model_metadata(
            config["disease_name"], 
            config, 
            features, 
            feature_types
        )
        
        # Get leaderboard if available
        leaderboard = None
        try:
            leaderboard = predictor.leaderboard().to_dict('records')
        except Exception as e:
            logging.warning(f"Could not get leaderboard: {e}")
        
        return {
            "csv_path": csv_path,
            "config_path": config_path,
            "config": config,
            "train_summary": train_summary,
            "features": features,
            "feature_types": feature_types,
            "leaderboard": leaderboard,
            "model_metadata_saved": str(input_fields_path),
            "disease_folder": f"models/{disease_name}/",
            "message": f"Model trained successfully for {config['disease_name']}"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
"""
# Add this endpoint to create/update the model registry
@app.post("/update-registry")
async def update_model_registry():
    """Update the model registry based on available trained models"""
    try:
        from utils import create_model_registry
        registry = create_model_registry()
        
        return {
            "message": "Model registry updated successfully",
            "models_found": len(registry),
            "models": list(registry.keys())
        }
    except Exception as e:
        logging.error(f"Error updating model registry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update registry: {e}")

# Updated upload and train endpoint to automatically update registry
@app.post("/upload-and-train")
async def upload_and_train(file: UploadFile = File(...)):
    print(f"✅ Received file: {file.filename}")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    contents = await file.read()
    csv_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(csv_path, "wb") as f:
        f.write(contents)

    try:
        # Generate config dict and save JSON
        config = generate_config_dict_from_csv(csv_path)
        config_name = os.path.splitext(file.filename)[0] + ".json"
        config_path = os.path.join(CONFIG_DIR, config_name)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Train the model
        train_summary = train_with_autogluon(config_path)
        
        # Load the trained predictor
        disease_name = config["disease_name"].replace(" ", "_").lower()
        predictor_path = f'models/{disease_name}/'
        predictor = TabularPredictor.load(predictor_path)
        
        # Get features and infer types from the original data
        features = predictor.feature_metadata.get_features()
        
        # Load original data to infer feature types
        df = pd.read_csv(csv_path)
        feature_types = infer_feature_types_from_sample(
            df, 
            config.get("categorical_columns", []), 
            config.get("numerical_columns", [])
        )
        
        # Save model metadata including input fields
        input_fields_path, config_path_saved = save_model_metadata(
            config["disease_name"], 
            config, 
            features, 
            feature_types
        )
        
        # Update the model registry automatically
        from utils import create_model_registry
        registry = create_model_registry()
        
        # Get leaderboard if available
        leaderboard = None
        try:
            leaderboard = predictor.leaderboard().to_dict('records')
        except Exception as e:
            logging.warning(f"Could not get leaderboard: {e}")
        
        return {
            "csv_path": csv_path,
            "config_path": config_path,
            "config": config,
            "train_summary": train_summary,
            "features": features,
            "feature_types": feature_types,
            "leaderboard": leaderboard,
            "model_metadata_saved": str(input_fields_path),
            "disease_folder": f"models/{disease_name}/",
            "registry_updated": True,
            "models_in_registry": len(registry),
            "message": f"Model trained successfully for {config['disease_name']} and registry updated"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    
# --- SUMMARY ENDPOINT ---
@app.post("/summary")
def summarize(req: SummaryRequest):
    try:
        summary = summarizer(req.raw_text, max_length=50, min_length=25, do_sample=False)
        return {
            "condition": req.condition_name,
            "summary": summary[0]["summary_text"]
        }
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

# --- MIDDLEWARE AND METRICS ---
from collections import Counter
import time

metrics = Counter()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    metrics['total_requests'] += 1
    logging.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000  # in ms
        logging.info(f"Response status: {response.status_code} | Latency: {process_time:.2f} ms")
        if response.status_code >= 400:
            metrics['errors'] += 1
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logging.error(f"Error during request: {request.method} {request.url} | Exception: {e} | Latency: {process_time:.2f} ms")
        metrics['errors'] += 1
        raise

# --- HEALTH CHECK AND METRICS ENDPOINTS ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "metrics": dict(metrics),
        "available_models": len(get_available_models())
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "metrics": dict(metrics),
        "available_models": get_available_models()
    }