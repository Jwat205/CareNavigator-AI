import json
from autogluon.tabular import TabularPredictor
import os
from pathlib import Path
import logging

# Get the base directory and models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = Path(BASE_DIR) / "models"

def load_model_and_features(disease_key: str):
    """
    Load model and features for a given disease.
    Now integrated with the training pipeline's folder structure.
    """
    # If disease_key is a set, extract the single element
    if isinstance(disease_key, set):
        disease_key = list(disease_key)[0]
    
    # Normalize disease name to match folder structure
    disease_folder_name = disease_key.replace(" ", "_").lower()
    disease_folder = models_dir / disease_folder_name
    
    logging.info(f"🔍 Looking for disease model: {disease_key}")
    logging.info(f"🔍 Disease folder path: {disease_folder}")
    
    # Check if disease folder exists
    if not disease_folder.exists():
        # Get available models for better error message
        available_models = []
        if models_dir.exists():
            available_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
        raise ValueError(f"Disease model folder not found for: {disease_key}. Available models: {available_models}")
    
    # Load the config file from the disease folder
    config_path = disease_folder / "config.json"
    input_fields_path = disease_folder / "input_fields.json"
    
    logging.info(f"🔍 Config path: {config_path}")
    logging.info(f"🔍 Input fields path: {input_fields_path}")
    
    # Check for required files
    if not config_path.exists():
        raise ValueError(f"Config file not found for disease: {disease_key} at {config_path}")
    
    if not input_fields_path.exists():
        raise ValueError(f"Input fields file not found for disease: {disease_key} at {input_fields_path}")
    
    # Load config and input fields
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.info(f"✅ Loaded config file")
    except Exception as e:
        raise ValueError(f"Failed to load config file for {disease_key}: {e}")
    
    try:
        with open(input_fields_path, "r") as f:
            input_fields_data = json.load(f)
        logging.info(f"✅ Loaded input fields file")
    except Exception as e:
        raise ValueError(f"Failed to load input fields file for {disease_key}: {e}")
    
    # Load the AutoGluon model - Try multiple possible paths
    model = None
    model_path_attempts = [
        disease_folder,  # AutoGluon saves to the folder directly
        disease_folder / "predictor.pkl",
        disease_folder / "models",
    ]
    
    for attempt_path in model_path_attempts:
        try:
            logging.info(f"🔍 Attempting to load model from: {attempt_path}")
            model = TabularPredictor.load(str(attempt_path))
            logging.info(f"✅ Successfully loaded model from: {attempt_path}")
            break
        except Exception as e:
            logging.warning(f"❌ Failed to load from {attempt_path}: {e}")
            continue
    
    if model is None:
        raise ValueError(f"Could not load AutoGluon model for {disease_key}. Tried paths: {model_path_attempts}")
    
    # Get all features from the model - Handle potential errors
    try:
        all_columns = model.feature_metadata.get_features()
        logging.info(f"✅ Got features from model metadata: {len(all_columns)} features")
    except Exception as e:
        logging.warning(f"⚠️ Could not get features from model metadata: {e}")
        # Fallback: try to get from input_fields.json
        all_columns = input_fields_data.get("features", [])
        if not all_columns:
            raise ValueError(f"Could not determine model features for {disease_key}")
        logging.info(f"✅ Using features from input_fields.json: {len(all_columns)} features")
    
    # Remove drop columns if specified in config
    drop_columns = config.get('drop_columns', [])
    # Always add "Unnamed: 0" to drop columns if not already there
    if "Unnamed: 0" not in drop_columns:
        drop_columns.append("Unnamed: 0")
    
    input_columns = [col for col in all_columns if col not in drop_columns]
    
    logging.info(f"✅ Model loaded for {disease_key}")
    logging.info(f"📊 Total features: {len(all_columns)}")
    logging.info(f"📊 Input features (after dropping): {len(input_columns)}")
    logging.info(f"🗂️ Drop columns: {drop_columns}")
    
    return model, input_columns

def get_model_metadata(disease_key: str):
    """
    Get complete metadata for a disease model including config and input fields info.
    """
    # Normalize disease name
    disease_folder_name = disease_key.replace(" ", "_").lower()
    disease_folder = models_dir / disease_folder_name
    
    if not disease_folder.exists():
        raise ValueError(f"Disease model folder not found for: {disease_key}")
    
    metadata = {}
    
    # Load config
    config_path = disease_folder / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                metadata["config"] = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load config for {disease_key}: {e}")
            metadata["config"] = {}
    
    # Load input fields
    input_fields_path = disease_folder / "input_fields.json"
    if input_fields_path.exists():
        try:
            with open(input_fields_path, "r") as f:
                metadata["input_fields"] = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load input fields for {disease_key}: {e}")
            metadata["input_fields"] = {}
    
    return metadata

def validate_prediction_inputs(disease_key: str, inputs: dict):
    """
    Validate and prepare inputs for prediction based on the model's requirements.
    Enhanced error handling and type conversion.
    """
    try:
        logging.info(f"🔍 Starting input validation for {disease_key}")
        logging.info(f"🔍 Raw inputs received: {inputs}")
        
        model, expected_inputs = load_model_and_features(disease_key)
        metadata = get_model_metadata(disease_key)
        
        # Get feature types if available
        feature_types = metadata.get("input_fields", {}).get("feature_types", {})
        config = metadata.get("config", {})
        
        logging.info(f"✅ Expected inputs: {len(expected_inputs)} features")
        logging.info(f"✅ First 5 expected: {expected_inputs[:5]}")
        
        # Prepare the input row
        row = {}
        missing_features = []
        
        for col in expected_inputs:
            if col in inputs and inputs[col] is not None and inputs[col] != "":
                value = inputs[col]
                
                # Enhanced type conversion
                try:
                    # Type conversion based on metadata
                    if col in feature_types:
                        ft = feature_types[col]
                        if ft.get("type") == "numerical":
                            row[col] = float(value) if value != "" else 0.0
                        else:
                            row[col] = str(value) if value is not None else ""
                    else:
                        # Smart type inference
                        if isinstance(value, (int, float)):
                            row[col] = float(value)
                        elif isinstance(value, str):
                            if value.replace('.', '').replace('-', '').replace('+', '').isdigit():
                                row[col] = float(value)
                            else:
                                row[col] = value
                        else:
                            row[col] = value
                            
                    logging.debug(f"✅ {col}: {inputs[col]} -> {row[col]} (type: {type(row[col])})")
                    
                except Exception as conv_error:
                    logging.warning(f"⚠️ Type conversion failed for {col}: {conv_error}")
                    # Fallback to original value
                    row[col] = value
                    
            else:
                # Handle missing values with smart defaults
                missing_features.append(col)
                
                # Smart default assignment
                default_value = get_smart_default(col, config)
                row[col] = default_value
                
                logging.debug(f"📝 Missing {col}: filled with default {default_value}")
        
        logging.info(f"✅ Input validation complete")
        logging.info(f"📊 Final row: {len(row)} features")
        logging.info(f"📝 Missing features filled: {len(missing_features)}")
        
        return row, missing_features, expected_inputs
        
    except Exception as e:
        logging.error(f"❌ Error validating inputs for {disease_key}: {e}")
        raise ValueError(f"Error validating inputs for {disease_key}: {str(e)}")

def get_smart_default(column_name: str, config: dict = None):
    """
    Get smart default values for missing features based on column name patterns.
    """
    col_lower = column_name.lower()
    
    # Handle special cases first
    if column_name == "Unnamed: 0":
        return 0
    
    # Location/text fields
    if any(keyword in col_lower for keyword in ["hospital", "county", "location", "name", "address"]):
        return ""
    
    # Age-related
    if "age" in col_lower:
        return 50
    
    # Binary features (common medical binary indicators)
    binary_indicators = [
        "sex", "male", "female", "fbs", "exang", "smoker", "diabetes", 
        "hypertension", "heart_disease", "stroke", "married", "employment"
    ]
    if any(indicator in col_lower for indicator in binary_indicators):
        return 0
    
    # Blood pressure
    if any(bp_term in col_lower for bp_term in ["bp", "pressure", "trestbps", "restbp"]):
        return 120
    
    # Cholesterol
    if any(chol_term in col_lower for chol_term in ["chol", "cholesterol"]):
        return 200
    
    # Heart rate
    if any(hr_term in col_lower for hr_term in ["thalach", "heartrate", "heart_rate", "pulse"]):
        return 150
    
    # BMI
    if "bmi" in col_lower:
        return 25.0
    
    # Glucose
    if "glucose" in col_lower:
        return 100
    
    # Categorical features with known ranges
    categorical_defaults = {
        "cp": 0,  # chest pain type
        "restecg": 0,  # rest ECG
        "slope": 1,  # slope of peak exercise ST segment
        "ca": 0,  # number of major vessels
        "thal": 2,  # thalassemia
        "work_type": "",
        "residence_type": "",
        "smoking_status": ""
    }
    
    if col_lower in categorical_defaults:
        return categorical_defaults[col_lower]
    
    # Check if it's marked as categorical in config
    if config and col_lower in config.get("categorical_columns", []):
        return ""
    
    # Float features
    if any(float_term in col_lower for float_term in ["oldpeak", "depression", "pedigree"]):
        return 0.0
    
    # Default to 0 for unknown numerical features
    return 0

def create_model_registry():
    """
    Create or update a model registry file based on available trained models.
    This bridges the gap between the old registry system and new folder structure.
    """
    registry = {}
    
    if not models_dir.exists():
        logging.warning(f"Models directory does not exist: {models_dir}")
        return registry
    
    logging.info(f"🔍 Scanning models directory: {models_dir}")
    
    for disease_dir in models_dir.iterdir():
        if disease_dir.is_dir():
            logging.info(f"🔍 Processing directory: {disease_dir.name}")
            
            config_file = disease_dir / "config.json"
            fields_file = disease_dir / "input_fields.json"
            
            if config_file.exists() and fields_file.exists():
                try:
                    # Load config and fields
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    
                    with open(fields_file, "r") as f:
                        fields_data = json.load(f)
                    
                    # Create registry entry
                    disease_name = fields_data.get("disease_name", disease_dir.name)
                    registry[disease_name] = {
                        "path": str(disease_dir),
                        "folder_name": disease_dir.name,
                        "drop_columns": config.get("drop_columns", []),
                        "target_column": config.get("target_column"),
                        "features": fields_data.get("features", []),
                        "feature_types": fields_data.get("feature_types", {}),
                        "created_at": fields_data.get("created_at")
                    }
                    
                    logging.info(f"✅ Added {disease_name} to registry")
                    
                except Exception as e:
                    logging.warning(f"❌ Error processing {disease_dir.name}: {e}")
            else:
                missing_files = []
                if not config_file.exists():
                    missing_files.append("config.json")
                if not fields_file.exists():
                    missing_files.append("input_fields.json")
                logging.warning(f"⚠️ {disease_dir.name} missing files: {missing_files}")
    
    # Save registry file
    registry_path = Path(BASE_DIR) / "model_registry.json"
    try:
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)
        logging.info(f"✅ Created model registry with {len(registry)} models at {registry_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save model registry: {e}")
    
    return registry

def get_available_models():
    """
    Get list of available models with their status.
    """
    available_models = []
    
    if not models_dir.exists():
        return available_models
    
    for disease_dir in models_dir.iterdir():
        if disease_dir.is_dir():
            config_file = disease_dir / "config.json"
            fields_file = disease_dir / "input_fields.json"
            
            # Check for AutoGluon model files
            has_autogluon = any(disease_dir.glob("**/*.pkl")) or any(disease_dir.glob("**/*.json"))
            
            model_info = {
                "disease_name": disease_dir.name.replace("_", " ").title(),
                "folder_name": disease_dir.name,
                "has_config": config_file.exists(),
                "has_input_fields": fields_file.exists(),
                "has_model_files": has_autogluon,
                "valid": config_file.exists() and fields_file.exists() and has_autogluon
            }
            
            if fields_file.exists():
                try:
                    with open(fields_file, "r") as f:
                        fields_data = json.load(f)
                    model_info.update({
                        "disease_name": fields_data.get("disease_name", model_info["disease_name"]),
                        "features_count": len(fields_data.get("features", [])),
                        "created_at": fields_data.get("created_at")
                    })
                except Exception:
                    pass
            
            available_models.append(model_info)
    
    return sorted(available_models, key=lambda x: x["disease_name"])

# Test function for debugging
def test_model_loading(disease_name: str = None):
    """
    Test model loading for debugging purposes.
    """
    if disease_name is None:
        # Test all available models
        available = get_available_models()
        if not available:
            print("❌ No models available to test")
            return
        
        for model_info in available:
            if model_info["valid"]:
                print(f"\n🧪 Testing model: {model_info['folder_name']}")
                test_model_loading(model_info["folder_name"])
                break
        return
    
    try:
        print(f"🧪 Testing model loading for: {disease_name}")
        model, features = load_model_and_features(disease_name)
        print(f"✅ Model loaded successfully")
        print(f"📊 Features: {len(features)}")
        
        # Test prediction with sample data
        sample_data = {feature: 0 for feature in features[:10]}  # Use first 10 features
        if "age" in features:
            sample_data["age"] = 45
        if "sex" in features:
            sample_data["sex"] = 1
            
        print(f"🧪 Testing prediction with sample data...")
        row, missing, expected = validate_prediction_inputs(disease_name, sample_data)
        print(f"✅ Input validation successful")
        print(f"📊 Prepared row: {len(row)} features")
        print(f"📝 Missing features: {len(missing)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed for {disease_name}: {e}")
        return False

if __name__ == "__main__":
    # Run tests when script is executed directly
    print("🧪 Testing utils.py functions...")
    
    # Test available models
    models = get_available_models()
    print(f"📊 Available models: {len(models)}")
    for model in models:
        print(f"   - {model['disease_name']} ({'✅' if model['valid'] else '❌'})")
    
    # Test model loading
    if models:
        test_model_loading()
    
    # Create registry
    registry = create_model_registry()
    print(f"📊 Registry entries: {len(registry)}")