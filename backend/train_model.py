import argparse
import json
import logging
import os
import pandas as pd
import joblib
from autogluon.tabular import TabularPredictor
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def create_input_fields_json(config, df, output_dir):
    target_column = config["target_column"]
    feature_columns = [col for col in df.columns if col != target_column]

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

    input_fields_path = os.path.join(output_dir, "input_fields.json")
    with open(input_fields_path, "w") as f:
        json.dump(input_fields_data, f, indent=2)

    logging.info(f"Created input_fields.json with {len(feature_columns)} features")
    return input_fields_path

def train_with_autogluon(config_path):
    config = load_config(config_path)
    df = pd.read_csv(config["data_path"])

    logging.info(f"Original data shape: {df.shape}")
    logging.info(f"Original columns: {list(df.columns)}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        logging.info("Dropped 'Unnamed: 0' column")

    for col in config.get("drop_columns", []):
        if col in df.columns:
            df = df.drop(columns=[col])
            logging.info(f"Dropped '{col}' column")

    if config["target_column"] == "readmitted":
        df["readmitted"] = df["readmitted"].apply(lambda val: 0 if val == "NO" else 1)
        logging.info("Converted 'readmitted' to binary")

    logging.info(f"Final data shape: {df.shape}")
    logging.info(f"Target distribution:\n{df[config['target_column']].value_counts().to_string()}")

    output_dir = config.get("output_dir", "models")
    os.makedirs(output_dir, exist_ok=True)

    predictor_path = os.path.join(output_dir, config["disease_name"].replace(" ", "_").lower())
    os.makedirs(predictor_path, exist_ok=True)

    input_fields_path = create_input_fields_json(config, df, predictor_path)
    logging.info(f"Input fields saved to: {input_fields_path}")

    logging.info("Starting AutoML with AutoGluon...")
    predictor = TabularPredictor(
        label=config["target_column"],
        path=predictor_path,
        eval_metric='roc_auc'
    ).fit(
        train_data=df,
        time_limit=config.get("time_limit", 600),
        presets=config.get("presets", "best_quality")
    )

    logging.info("Testing trained model...")
    try:
        sample_row = df.drop(columns=[config["target_column"]]).iloc[0:1]
        test_prediction = predictor.predict(sample_row)
        logging.info(f"Model test successful: {test_prediction.tolist()}")
        try:
            predictor.predict_proba(sample_row)
            logging.info("Probabilities test successful")
        except Exception as prob_error:
            logging.warning(f"Probabilities not available: {prob_error}")
    except Exception as test_error:
        logging.error(f"Model test failed: {test_error}")
        raise Exception(f"Trained model failed basic test: {test_error}")

    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.sort_values(by='score_val', ascending=False).iloc[0]['model']
    logging.info(f"Best model: {best_model}")

    feature_importance = predictor.feature_importance(df)
    logging.info(f"Model saved to: {predictor_path}")

    try:
        TabularPredictor.load(predictor_path)
        logging.info("Model reload verified")
    except Exception as reload_error:
        raise Exception(f"Model cannot be reloaded: {reload_error}")

    return {
        "leaderboard": leaderboard.to_dict(orient='records'),
        "feature_importance": feature_importance.head(10).to_dict(orient='records'),
        "model_path": predictor_path,
        "input_fields_path": input_fields_path,
        "target_counts": df[config["target_column"]].value_counts().to_dict(),
        "feature_count": len(df.columns) - 1,
        "data_shape": list(df.shape)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using AutoML with config file")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    try:
        result = train_with_autogluon(args.config)
        logging.info(f"TRAINING COMPLETE — model at {result['model_path']}, {result['feature_count']} features")
    except Exception as e:
        logging.error(f"TRAINING FAILED: {e}")
        raise
