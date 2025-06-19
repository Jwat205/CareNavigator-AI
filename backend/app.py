import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, Any, List

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CareNavigator AI Demo", layout="centered")

st.title("🩺 CareNavigator AI Demo")

# Helper functions
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_available_models():
    """Get list of available trained models from the backend"""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json().get("available_models", [])
        else:
            st.error(f"Error fetching models: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return []

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_model_metadata(disease_name: str):
    """Get metadata for a specific disease model"""
    try:
        response = requests.get(f"{API_URL}/models/{disease_name}/metadata")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching metadata for {disease_name}: {e}")
        return None

def render_input_field(field_name: str, field_info: Dict[str, Any], key_suffix: str = ""):
    """Render appropriate input field based on field type and metadata"""
    key = f"{field_name}_{key_suffix}"
    
    if field_info.get("type") == "categorical":
        options = field_info.get("options", [])
        if options:
            # Dropdown for categorical with known options
            return st.selectbox(
                f"{field_name}",
                options=[""] + options,  # Add empty option
                key=key,
                help=f"Select a value for {field_name}"
            )
        else:
            # Text input for categorical without known options
            return st.text_input(
                f"{field_name}",
                key=key,
                help=f"Enter categorical value for {field_name}"
            )
    
    elif field_info.get("type") == "numerical":
        # Number input for numerical fields
        min_val = field_info.get("min", 0)
        max_val = field_info.get("max", 100)
        mean_val = field_info.get("mean", (min_val + max_val) / 2)
        
        return st.number_input(
            f"{field_name}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(mean_val),
            key=key,
            help=f"Enter numerical value for {field_name} (range: {min_val} - {max_val})"
        )
    
    else:
        # Default to text input
        return st.text_input(
            f"{field_name}",
            key=key,
            help=f"Enter value for {field_name}"
        )

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Insurance Match", "Disease Risk Prediction", "Summary", "AutoML Upload & Train"])

with tab1:
    st.header("💼 Insurance Plan Matcher")
    st.markdown("Describe your situation, and we'll find the best-matching insurance plans for you.")

    desc = st.text_area(
        "📝 Description of Insurance Needs",
        placeholder="Example: I am a 45-year-old in Texas with diabetes who needs prescription coverage.",
        height=120
    )

    if st.button("🔍 Find Plans"):
        if not desc.strip():
            st.warning("⚠️ Please enter a description.")
        else:
            with st.spinner("🔄 Matching plans..."):
                try:
                    resp = requests.post(f"{API_URL}/insurance-match/", json={"description": desc})
                    if resp.status_code == 200:
                        data = resp.json()

                        if data["matched_plans"]:
                            st.success(f"✅ {len(data['matched_plans'])} Match(es) Found!")
                            
                            for match in data["detailed_matches"]:
                                with st.expander(f"📌 {match['plan_name']} (Score: {match['score']}/1.0)"):
                                    st.markdown(f"**📝 Description:** {match['description']}")
                                    st.markdown("**📊 Score Breakdown:**")
                                    st.json(match["score_breakdown"])
                                    
                                    if match["reasons"]:
                                        st.markdown("✅ **Why this plan matched:**")
                                        for r in match["reasons"]:
                                            st.markdown(f"- {r}")
                                    if match["warnings"]:
                                        st.markdown("⚠️ **Warnings or Limitations:**")
                                        for w in match["warnings"]:
                                            st.markdown(f"- {w}")

                                    st.markdown("🗺️ **Plan Details:**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"- **States Covered:** {', '.join(match['plan_details']['states'])}")
                                        st.markdown(f"- **Min Age:** {match['plan_details']['min_age']}")
                                    with col2:
                                        st.markdown(f"- **Max Age:** {match['plan_details']['max_age']}")
                                        st.markdown(f"- **Coverage:** {', '.join(match['plan_details']['coverage'])}")

                        else:
                            st.warning("😕 No suitable plans found.")
                        
                        # Explanation block
                        st.markdown("### 🧠 Matching Explanation")
                        st.info(data["explanation"])

                        # Profile Summary
                        with st.expander("🧑‍⚕️ View Your Profile Used for Matching"):
                            st.json(data["user_profile"])

                        st.caption(f"🔍 Matching algorithm: {data['matching_algorithm']}")
                        st.caption(f"📊 Total plans evaluated: {data['total_plans_evaluated']}")
                    
                    else:
                        st.error(f"❌ Error: {resp.text}")
                except Exception as e:
                    st.error(f"🚨 Connection error: {e}")

# Tab 2: Disease Risk Prediction - Enhanced with dynamic field loading
with tab2:
    st.header("Disease Risk Predictor")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.warning("No trained models available. Please train a model first in the AutoML tab.")
    else:
        # Model selection dropdown
        model_options = {model["disease_name"]: model["folder_name"] for model in available_models}
        
        selected_disease = st.selectbox(
            "Select Disease/Condition Model",
            options=list(model_options.keys()),
            help="Choose from available trained models"
        )
        
        if selected_disease:
            folder_name = model_options[selected_disease]
            
            # Get model metadata
            metadata = get_model_metadata(folder_name)
            
            if metadata and "input_fields" in metadata:
                input_fields_data = metadata["input_fields"]
                features = input_fields_data.get("features", [])
                feature_types = input_fields_data.get("feature_types", {})
                
                st.write(f"**Model Info:** {selected_disease}")
                st.write(f"**Features Required:** {len(features)}")
                
                if features:
                    st.subheader("Enter Feature Values")
                    
                    # Create input form
                    with st.form("prediction_form"):
                        input_values = {}
                        
                        # Create columns for better layout
                        col1, col2 = st.columns(2)
                        
                        for i, feature in enumerate(features):
                            field_info = feature_types.get(feature, {"type": "text"})
                            
                            # Alternate between columns for better layout
                            with col1 if i % 2 == 0 else col2:
                                value = render_input_field(feature, field_info, "pred")
                                if value is not None and value != "":
                                    input_values[feature] = value
                        
                        submitted = st.form_submit_button("🔮 Predict Risk")
                    
                    if submitted:
                        if not input_values:
                            st.warning("Please enter at least some feature values.")
                        else:
                            with st.spinner("Predicting..."):
                                try:
                                    payload = {"disease": folder_name, "inputs": input_values}
                                    resp = requests.post(f"{API_URL}/predict", json=payload)
                                    
                                    if resp.status_code == 200:
                                        result = resp.json()
                                        st.success("✅ Prediction Complete!")
                                        
                                        # Display results in a nice format
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.metric("Prediction", result["prediction"])
                                        
                                        with col2:
                                            if result.get("probabilities"):
                                                st.write("**Probabilities:**")
                                                for class_name, prob in result["probabilities"].items():
                                                    st.write(f"- {class_name}: {prob:.3f}")
                                        
                                        # Show input values used
                                        with st.expander("Input Values Used"):
                                            st.json(result.get("input_values", {}))
                                    
                                    else:
                                        st.error(f"Prediction failed: {resp.text}")
                                
                                except Exception as e:
                                    st.error(f"Error during prediction: {e}")
            else:
                st.error(f"Could not load metadata for {selected_disease}. The model may not be properly configured.")

# Tab 3: Summary
with tab3:
    st.header("Condition Summarizer")
    condition_name = st.text_input("Condition name for summary (e.g., diabetes)")
    raw_text = st.text_area("Paste medical/health info to summarize", height=120)
    
    if st.button("Summarize"):
        if not (condition_name and raw_text):
            st.warning("Please provide both condition name and text to summarize.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    resp = requests.post(f"{API_URL}/summary", json={"condition_name": condition_name, "raw_text": raw_text})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.write(f"**{data['condition']} summary:**")
                        st.write(data["summary"])
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# Tab 4: AutoML Upload & Train - Enhanced
with tab4:
    st.header("AutoML: Upload CSV, Train, and Predict")
    st.write("Upload your dataset, train a model, and the input fields will be automatically saved for future predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV file for AutoML", type="csv", key="upload-train")
    
    if st.button("Upload and Train"):
        if uploaded_file is None:
            st.warning("Please upload a CSV file.")
        else:
            with st.spinner("Uploading and training... (this may take a few minutes)"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                try:
                    resp = requests.post(f"{API_URL}/upload-and-train", files=files)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    st.success("✅ Training Complete!")
                    
                    # Display results in organized sections
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Model Info")
                        st.write(f"**Disease:** {data['config']['disease_name']}")
                        st.write(f"**Target:** {data['config']['target_column']}")
                        st.write(f"**Features:** {len(data['features'])}")
                        st.write(f"**Model Type:** {data['config']['model_type']}")
                    
                    with col2:
                        st.subheader("Files Created")
                        st.write(f"📄 CSV: `{data['csv_path']}`")
                        st.write(f"⚙️ Config: `{data['config_path']}`")
                        st.write(f"🤖 Model: `{data['disease_folder']}`")
                        st.write(f"📊 Metadata: `{data['model_metadata_saved']}`")
                    
                    # Training summary
                    if data.get("train_summary"):
                        with st.expander("Training Summary"):
                            st.json(data["train_summary"])
                    
                    # Leaderboard
                    if data.get("leaderboard"):
                        with st.expander("Model Leaderboard"):
                            df_leaderboard = pd.DataFrame(data["leaderboard"])
                            st.dataframe(df_leaderboard)
                    
                    # Feature types
                    if data.get("feature_types"):
                        with st.expander("Feature Types & Statistics"):
                            st.json(data["feature_types"])
                    
                    # Sample predictions
                    if data.get("sample_predictions"):
                        with st.expander("Sample Predictions"):
                            df_samples = pd.DataFrame(data["sample_predictions"])
                            st.dataframe(df_samples)
                    
                    # Clear cache to refresh available models
                    st.cache_data.clear()
                    
                    st.info("💡 Model is now available in the 'Disease Risk Prediction' tab!")
                    
                except requests.exceptions.HTTPError as e:
                    st.error(f"Training failed: {e.response.text if e.response else str(e)}")
                except Exception as e:
                    st.error(f"Error during training: {e}")
    
    # Section for manual model testing
    st.divider()
    st.subheader("Test Uploaded Model")
    st.write("Quick test your newly trained model before using it in the prediction tab.")
    
    # Simple input form for quick testing
    uploaded_test_file = st.file_uploader("Upload test CSV (optional)", type="csv", key="test-upload")
    
    if uploaded_test_file:
        test_data = pd.read_csv(uploaded_test_file)
        st.write("**Test Data Preview:**")
        st.dataframe(test_data.head())
        
        if st.button("Run Test Predictions"):
            with st.spinner("Running test predictions..."):
                try:
                    files = {"file": (uploaded_test_file.name, uploaded_test_file.getvalue(), "text/csv")}
                    resp = requests.post(f"{API_URL}/test-predictions", files=files)
                    
                    if resp.status_code == 200:
                        results = resp.json()
                        st.success("✅ Test predictions completed!")
                        
                        # Show results
                        if results.get("predictions"):
                            st.write("**Predictions:**")
                            df_results = pd.DataFrame(results["predictions"])
                            st.dataframe(df_results)
                        
                        if results.get("accuracy"):
                            st.metric("Accuracy", f"{results['accuracy']:.3f}")
                        
                        if results.get("metrics"):
                            st.write("**Performance Metrics:**")
                            st.json(results["metrics"])
                    else:
                        st.error(f"Test failed: {resp.text}")
                
                except Exception as e:
                    st.error(f"Error during testing: {e}")

# Sidebar - Model Management
with st.sidebar:
    st.header("Model Management")
    
    # Refresh models button
    if st.button("🔄 Refresh Models"):
        st.cache_data.clear()
        st.rerun()
    
    # Show available models
    models = get_available_models()
    if models:
        st.write("**Available Models:**")
        for model in models:
            st.write(f"• {model['disease_name']}")
    else:
        st.write("No models available")
    
    # Model deletion (if needed)
    st.divider()
    if models:
        st.subheader("Delete Model")
        model_to_delete = st.selectbox(
            "Select model to delete",
            options=[""] + [model["disease_name"] for model in models],
            key="delete_model"
        )
        
        if model_to_delete and st.button("🗑️ Delete Model", type="secondary"):
            try:
                folder_name = next(m["folder_name"] for m in models if m["disease_name"] == model_to_delete)
                resp = requests.delete(f"{API_URL}/models/{folder_name}")
                
                if resp.status_code == 200:
                    st.success(f"Model '{model_to_delete}' deleted successfully!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Failed to delete model: {resp.text}")
            
            except Exception as e:
                st.error(f"Error deleting model: {e}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>CareNavigator AI Demo | Powered by AutoML & Streamlit</p>
    <p>🔗 Backend API: <code>http://localhost:8000</code></p>
</div>
""", unsafe_allow_html=True)