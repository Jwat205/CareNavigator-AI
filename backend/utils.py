"""
def load_model_and_features(disease_key: str):

    # If disease_key is a set, extract the single element
    if isinstance(disease_key, set):
        disease_key = list(disease_key)[0]
    
    with open("model_registry.json") as f:
        registry = json.load(f)
    
    if disease_key not in registry:
        raise ValueError(f"Disease '{disease_key}' not found in registry")
    
    model_info = registry[disease_key]
    print(f"model_info['path']/{disease_key}")
    model = TabularPredictor.load(f"{model_info['path']}/{disease_key}")
    
    # Remove drop columns from full feature set
    all_columns = model.feature_metadata.get_features()
    input_columns = [col for col in all_columns if col not in model_info.get('drop_columns', [])]
    
    return model, input_columns
"""
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
    
    # Check if disease folder exists
    if not disease_folder.exists():
        raise ValueError(f"Disease model folder not found for: {disease_key}")
    
    # Load the config file from the disease folder
    config_path = disease_folder / "config.json"
    input_fields_path = disease_folder / "input_fields.json"
    
    if not config_path.exists():
        raise ValueError(f"Config file not found for disease: {disease_key}")
    
    if not input_fields_path.exists():
        raise ValueError(f"Input fields file not found for disease: {disease_key}")
    
    # Load config and input fields
    with open(config_path, "r") as f:
        config = json.load(f)
    
    with open(input_fields_path, "r") as f:
        input_fields_data = json.load(f)
    
    # Load the AutoGluon model
    model_path = disease_folder  # AutoGluon saves to the folder directly
    
    try:
        model = TabularPredictor.load(str(model_path))
        print(f"✅ Successfully loaded model from: {model_path}")
    except Exception as e:
        # Try alternative path structure
        try:
            model = TabularPredictor.load(str(disease_folder / "predictor.pkl"))
            print(f"✅ Successfully loaded model from: {disease_folder / 'predictor.pkl'}")
        except Exception as e2:
            raise ValueError(f"Could not load model for {disease_key}. Tried paths: {model_path} and {disease_folder / 'predictor.pkl'}. Errors: {e}, {e2}")
    
    # Get all features from the model
    all_columns = model.feature_metadata.get_features()
    
    # Remove drop columns if specified in config
    drop_columns = config.get('drop_columns', [])
    input_columns = [col for col in all_columns if col not in drop_columns]
    
    print(f"✅ Model loaded for {disease_key}")
    print(f"📊 Total features: {len(all_columns)}")
    print(f"📊 Input features (after dropping): {len(input_columns)}")
    print(f"🗂️ Drop columns: {drop_columns}")
    
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
        with open(config_path, "r") as f:
            metadata["config"] = json.load(f)
    
    # Load input fields
    input_fields_path = disease_folder / "input_fields.json"
    if input_fields_path.exists():
        with open(input_fields_path, "r") as f:
            metadata["input_fields"] = json.load(f)
    
    return metadata

def validate_prediction_inputs(disease_key: str, inputs: dict):
    """
    Validate and prepare inputs for prediction based on the model's requirements.
    """
    try:
        model, expected_inputs = load_model_and_features(disease_key)
        metadata = get_model_metadata(disease_key)
        
        # Get feature types if available
        feature_types = metadata.get("input_fields", {}).get("feature_types", {})
        config = metadata.get("config", {})
        
        # Prepare the input row
        row = {}
        missing_features = []
        
        for col in expected_inputs:
            if col in inputs and inputs[col] is not None:
                value = inputs[col]
                
                # Type conversion based on metadata
                if col in feature_types:
                    ft = feature_types[col]
                    if ft.get("type") == "numerical":
                        try:
                            row[col] = float(value) if value != "" else 0.0
                        except (ValueError, TypeError):
                            row[col] = 0.0
                    else:
                        row[col] = str(value) if value is not None else ""
                else:
                    # Fallback type inference
                    try:
                        if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                            row[col] = float(value)
                        else:
                            row[col] = value
                    except:
                        row[col] = value
            else:
                # Handle missing values with smart defaults
                missing_features.append(col)
                
                if col == "Unnamed: 0":
                    row[col] = 0
                elif any(keyword in col.lower() for keyword in ["hospital", "county", "location", "name"]):
                    row[col] = ""
                elif col in config.get("categorical_columns", []):
                    row[col] = ""
                else:
                    row[col] = 0
        
        return row, missing_features, expected_inputs
        
    except Exception as e:
        raise ValueError(f"Error validating inputs for {disease_key}: {e}")

def create_model_registry():
    """
    Create or update a model registry file based on available trained models.
    This bridges the gap between the old registry system and new folder structure.
    """
    registry = {}
    
    if not models_dir.exists():
        return registry
    
    for disease_dir in models_dir.iterdir():
        if disease_dir.is_dir():
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
                    
                except Exception as e:
                    logging.warning(f"Error processing {disease_dir.name}: {e}")
    
    # Save registry file
    registry_path = Path(BASE_DIR) / "model_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)
    
    print(f"✅ Created model registry with {len(registry)} models")
    return registry