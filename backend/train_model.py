import argparse
import json
import os
import pandas as pd
import joblib
from autogluon.tabular import TabularPredictor
from datetime import datetime

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def create_input_fields_json(config, df, output_dir):
    """Create input_fields.json for the API to use"""
    
    # Get all feature columns (exclude target)
    target_column = config["target_column"]
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Remove any columns that were dropped
    drop_columns = config.get("drop_columns", [])
    if "Unnamed: 0" not in drop_columns:
        drop_columns.append("Unnamed: 0")
    
    feature_columns = [col for col in feature_columns if col not in drop_columns]
    
    input_fields_data = {
        "disease_name": config["disease_name"],
        "features": feature_columns,
        "target_column": target_column,
        "model_type": "autogluon",
        "created_at": datetime.now().isoformat(),
        "data_shape": list(df.shape),
        "feature_count": len(feature_columns),
        "description": f"Input fields for {config['disease_name']} prediction model"
    }
    
    # Save input_fields.json
    input_fields_path = os.path.join(output_dir, "input_fields.json")
    with open(input_fields_path, "w") as f:
        json.dump(input_fields_data, f, indent=2)
    
    print(f"✅ Created input_fields.json with {len(feature_columns)} features")
    return input_fields_path

def train_with_autogluon(config_path):
    config = load_config(config_path)
    df = pd.read_csv(config["data_path"])
    
    print(f"📊 Original data shape: {df.shape}")
    print(f"📊 Original columns: {list(df.columns)}")

    # Always drop Unnamed: 0 if it exists
    if "Unnamed: 0" in df.columns:
        print("🧹 Dropping 'Unnamed: 0' column from data!")
        df = df.drop(columns=["Unnamed: 0"])

    # Drop other columns listed in config['drop_columns']
    for col in config.get("drop_columns", []):
        if col in df.columns:
            print(f"🧹 Dropping '{col}' column from data!")
            df = df.drop(columns=[col])

    if config["target_column"] == "readmitted":
        print("\n🔄 Converting 'readmitted' column to binary (NO = 0, <30/>30 = 1)")
        df["readmitted"] = df["readmitted"].apply(lambda val: 0 if val == "NO" else 1)

    print(f"\n📊 Final data shape: {df.shape}")
    print(f"📊 Final columns: {list(df.columns)}")
    print("\n🎯 Target value counts before training:")
    print(df[config["target_column"]].value_counts())

    output_dir = config.get("output_dir", "models")
    os.makedirs(output_dir, exist_ok=True)
        
    # Create a specific path for this predictor
    predictor_path = os.path.join(output_dir, "{disease_name}".format(disease_name=config["disease_name"].replace(" ", "_").lower()))
    os.makedirs(predictor_path, exist_ok=True)

    # Create input_fields.json BEFORE training
    input_fields_path = create_input_fields_json(config, df, predictor_path)
    print(f"📝 Input fields saved to: {input_fields_path}")

    print(f"\n🚀 Starting AutoML with AutoGluon...")
    # Specify the path in the constructor - AutoGluon will save here automatically
    predictor = TabularPredictor(
        label=config["target_column"], 
        path=predictor_path,
        eval_metric='roc_auc'
    ).fit(
        train_data=df,
        time_limit=config.get("time_limit", 600),  # 10 min default
        presets=config.get("presets", "best_quality")
    )

    # Test the model immediately after training
    print(f"\n🧪 Testing trained model...")
    try:
        # Create sample prediction to verify model works
        sample_row = df.drop(columns=[config["target_column"]]).iloc[0:1]
        test_prediction = predictor.predict(sample_row)
        print(f"✅ Model test successful: {test_prediction}")
        
        # Test probabilities
        try:
            test_probabilities = predictor.predict_proba(sample_row)
            print(f"✅ Probabilities test successful")
        except Exception as prob_error:
            print(f"⚠️ Probabilities not available: {prob_error}")
            
    except Exception as test_error:
        print(f"❌ Model test failed: {test_error}")
        raise Exception(f"Trained model failed basic test: {test_error}")

    print("\n✅ Leaderboard:")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard)

    best_model_name = leaderboard.sort_values(by='score_val', ascending=False).iloc[0]['model']
    print(f"🏆 Best model: {best_model_name}")
        
    print("\n📊 Top Features:")
    feature_importance = predictor.feature_importance(df)
    print(feature_importance.head(10))

    print(f"\n✅ AutoGluon model saved in: {predictor_path}")
    print(f"✅ Input fields saved in: {input_fields_path}")

    # Verify model can be reloaded
    print(f"\n🔄 Verifying model reload...")
    try:
        reloaded_predictor = TabularPredictor.load(predictor_path)
        print(f"✅ Model reload successful")
    except Exception as reload_error:
        print(f"❌ Model reload failed: {reload_error}")
        raise Exception(f"Model cannot be reloaded: {reload_error}")

    leaderboard_dict = leaderboard.to_dict(orient='records')
    feature_importance_dict = feature_importance.head(10).to_dict(orient='records')
    
    return {
        "leaderboard": leaderboard_dict,
        "feature_importance": feature_importance_dict,
        "model_path": predictor_path,  # Return the actual path where predictor is saved
        "input_fields_path": input_fields_path,
        "target_counts": df[config["target_column"]].value_counts().to_dict(),
        "feature_count": len(df.columns) - 1,  # Exclude target
        "data_shape": list(df.shape)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using AutoML with config file")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    try:
        result = train_with_autogluon(args.config)
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"Model saved to: {result['model_path']}")
        print(f"Input fields saved to: {result['input_fields_path']}")
        print(f"Features: {result['feature_count']}")
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        raise