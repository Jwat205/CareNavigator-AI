import argparse
import json
import os
import pandas as pd
import joblib
from autogluon.tabular import TabularPredictor

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def train_with_autogluon(config_path):
    config = load_config(config_path)
    df = pd.read_csv(config["data_path"])

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

    print("\n🎯 Target value counts before training:")
    print(df[config["target_column"]].value_counts())

    output_dir = config.get("output_dir", "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a specific path for this predictor
    predictor_path = os.path.join(output_dir, "{disease_name}".format(disease_name=config["disease_name"].replace(" ", "_").lower()))
    os.makedirs(predictor_path, exist_ok=True)

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

    # No need to call predictor.save() - it's already saved in the path specified above

    print("\n✅ Leaderboard:")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard)

    best_model_name = leaderboard.sort_values(by='score_val', ascending=False).iloc[0]['model']
    print(f"Best model: {best_model_name}")
    
    print("\n📊 Top Features:")
    feature_importance = predictor.feature_importance(df)
    print(feature_importance.head(10))

    model_path = os.path.join(output_dir, config["disease_name"].replace(" ", "_").lower())
    print(f"\n✅ AutoGluon model saved in: {predictor_path}")

    leaderboard_dict = leaderboard.to_dict(orient='records')
    feature_importance_dict = feature_importance.head(10).to_dict(orient='records')
    return {
        "leaderboard": leaderboard_dict,
        "feature_importance": feature_importance_dict,
        "model_path": predictor_path,  # Return the actual path where predictor is saved
        "target_counts": df[config["target_column"]].value_counts().to_dict(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model using AutoML with config file")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    train_with_autogluon(args.config)