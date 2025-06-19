import pandas as pd
import argparse
import os
import json
from openai import OpenAI

def generate_prompt(df_sample_csv):
    return f"""
You are an expert AutoML assistant.

Given the following CSV sample, generate a **minimal and efficient JSON config** for training a predictive model from hospital-provided data.

Instructions:
- "disease_name": infer from the dataset (e.g., "diabetes", "heart_failure")
- "target_column": select a binary label column to predict (e.g., "readmitted", "has_disease")
- "drop_columns": include IDs, redundant codes, or administrative fields not useful for learning
- "categorical_columns": columns with discrete non-numeric categories (✔️ include target_column if categorical)
- "numerical_columns": continuous or count-based features (✔️ include target_column if numeric)
- "model_type": pick either "XGBoost" or "LogisticRegression"
- "model_path": path to save the trained model (e.g., "models/diabetes")

⚠️ REQUIREMENTS:
- The target_column **must be included** in either `categorical_columns` or `numerical_columns`, based on its type.
- DO NOT include ID-like columns (e.g., anything ending in `_id`, `patient_nbr`, etc.) unless critical.
- Minimize noise. Only keep Tier 1 features likely to generalize across patients.
- Output must be **valid JSON** parsable by `json.loads()`.

Only return the JSON config — no extra explanation or markdown.

CSV SAMPLE:
{df_sample_csv}
"""

def request_config_from_gpt(csv_path, output_path):
    # Read CSV and prepare prompt
    df = pd.read_csv(csv_path)
    sample = df.head(20).to_csv(index=False)
    prompt = generate_prompt(sample)

    print("\n[INFO] Sending prompt to GPT-4...\n")

    # Authenticate
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY environment variable not set!")

    client = OpenAI(api_key=api_key)

    # Send prompt to GPT
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    config_json = response.choices[0].message.content.strip()
    print("\n----- RAW LLM OUTPUT -----\n")
    print(config_json)
    print("\n--------------------------\n")

    # Remove common markdown wrappers
    if config_json.startswith("```"):
        config_json = config_json.lstrip("`").strip()
        # Remove leading language specifier (e.g., "json")
        if config_json.lower().startswith("json"):
            config_json = config_json[4:].strip()
    # Remove any trailing triple backticks
    config_json = config_json.rstrip("`").strip()

    # Check for empty response
    if not config_json:
        print("❌ ERROR: LLM output was empty!")
        print("PROMPT SENT:\n", prompt)
        exit(1)

    # Try parsing, print if error
    try:
        parsed = json.loads(config_json)
    except Exception as e:
        print("❌ ERROR: LLM returned invalid JSON!")
        print("PROMPT SENT:\n", prompt)
        print("RAW OUTPUT:\n", config_json)
        raise e

    # Add path for train_model.py
    parsed["data_path"] = csv_path

    # Save to file
    with open(output_path, "w") as f:
        json.dump(parsed, f, indent=4)
    print(f"\n✅ Config saved to: {output_path}\n")

def generate_config_dict_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    sample = df.head(20).to_csv(index=False)
    prompt = generate_prompt(sample)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY environment variable not set!")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    config_json = response.choices[0].message.content.strip()

    if config_json.startswith("```"):
        config_json = config_json.lstrip("`").strip()
        if config_json.lower().startswith("json"):
            config_json = config_json[4:].strip()
    config_json = config_json.rstrip("`").strip()

    if not config_json:
        raise ValueError("❌ LLM output was empty!")

    try:
        parsed = json.loads(config_json)
    except Exception as e:
        raise ValueError(f"❌ LLM returned invalid JSON! RAW OUTPUT:\n{config_json}") from e

    parsed["data_path"] = csv_path
    return parsed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-generate config JSON from disease dataset.")
    parser.add_argument("--csv", required=True, help="Path to disease dataset CSV")
    parser.add_argument("--out", required=True, help="Path to output config JSON file")
    args = parser.parse_args()

    request_config_from_gpt(args.csv, args.out)
