import json

import pytest

import utils


class TestGetSmartDefault:
    def test_unnamed_index_column(self):
        assert utils.get_smart_default("Unnamed: 0") == 0

    def test_location_text_field_defaults_to_empty_string(self):
        assert utils.get_smart_default("hospital_name") == ""

    def test_age_field(self):
        assert utils.get_smart_default("Age") == 50

    def test_binary_indicator_field(self):
        assert utils.get_smart_default("smoker") == 0

    def test_blood_pressure_field(self):
        assert utils.get_smart_default("trestbps") == 120

    def test_cholesterol_field(self):
        assert utils.get_smart_default("chol") == 200

    def test_heart_rate_field(self):
        assert utils.get_smart_default("thalach") == 150

    def test_bmi_field(self):
        assert utils.get_smart_default("bmi") == 25.0

    def test_glucose_field(self):
        assert utils.get_smart_default("glucose") == 100

    def test_known_categorical_field(self):
        assert utils.get_smart_default("thal") == 2

    def test_config_marked_categorical_field(self):
        config = {"categorical_columns": ["custom_field"]}
        assert utils.get_smart_default("custom_field", config) == ""

    def test_float_field(self):
        assert utils.get_smart_default("oldpeak") == 0.0

    def test_unknown_field_defaults_to_zero(self):
        assert utils.get_smart_default("totally_unrecognized_column") == 0


class TestLoadModelAndFeatures:
    def test_raises_for_missing_disease_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path)
        with pytest.raises(ValueError, match="Disease model folder not found"):
            utils.load_model_and_features("nonexistent_disease")

    def test_accepts_a_set_with_a_single_disease_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path)
        with pytest.raises(ValueError, match="diabetes"):
            utils.load_model_and_features({"diabetes"})

    def test_raises_for_missing_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path)
        disease_dir = tmp_path / "diabetes"
        disease_dir.mkdir()
        (disease_dir / "input_fields.json").write_text("{}")
        with pytest.raises(ValueError, match="Config file not found"):
            utils.load_model_and_features("diabetes")

    def test_raises_for_missing_input_fields_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path)
        disease_dir = tmp_path / "diabetes"
        disease_dir.mkdir()
        (disease_dir / "config.json").write_text("{}")
        with pytest.raises(ValueError, match="Input fields file not found"):
            utils.load_model_and_features("diabetes")


class TestValidatePredictionInputs:
    def _make_model(self, monkeypatch, features, feature_types=None, config=None, drop_columns=None):
        class FakeModel:
            def predict(self, df):
                return [1]

        monkeypatch.setattr(
            utils, "load_model_and_features", lambda disease_key: (FakeModel(), features)
        )
        monkeypatch.setattr(
            utils,
            "get_model_metadata",
            lambda disease_key: {
                "input_fields": {"feature_types": feature_types or {}},
                "config": config or {},
            },
        )

    def test_fills_missing_features_with_smart_defaults(self, monkeypatch):
        self._make_model(monkeypatch, features=["age", "bmi"])
        row, missing, expected = utils.validate_prediction_inputs("diabetes", {"age": 40})

        assert row["age"] == 40.0
        assert row["bmi"] == 25.0
        assert missing == ["bmi"]
        assert expected == ["age", "bmi"]

    def test_strict_mode_raises_when_features_missing(self, monkeypatch):
        self._make_model(monkeypatch, features=["age", "bmi"])
        with pytest.raises(utils.StrictValidationError) as exc_info:
            utils.validate_prediction_inputs("diabetes", {"age": 40}, strict=True)
        assert exc_info.value.missing_fields == ["bmi"]

    def test_strict_mode_passes_when_all_features_present(self, monkeypatch):
        self._make_model(monkeypatch, features=["age", "bmi"])
        row, missing, _ = utils.validate_prediction_inputs(
            "diabetes", {"age": 40, "bmi": 22.5}, strict=True
        )
        assert missing == []
        assert row["bmi"] == 22.5

    def test_numerical_feature_type_conversion(self, monkeypatch):
        self._make_model(
            monkeypatch,
            features=["age"],
            feature_types={"age": {"type": "numerical"}},
        )
        row, _, _ = utils.validate_prediction_inputs("diabetes", {"age": "40"})
        assert row["age"] == 40.0
        assert isinstance(row["age"], float)

    def test_categorical_feature_type_kept_as_string(self, monkeypatch):
        self._make_model(
            monkeypatch,
            features=["smoking_status"],
            feature_types={"smoking_status": {"type": "categorical"}},
        )
        row, _, _ = utils.validate_prediction_inputs("diabetes", {"smoking_status": "former"})
        assert row["smoking_status"] == "former"

    def test_empty_string_input_treated_as_missing(self, monkeypatch):
        self._make_model(monkeypatch, features=["age"])
        row, missing, _ = utils.validate_prediction_inputs("diabetes", {"age": ""})
        assert missing == ["age"]
        assert row["age"] == 50  # smart default for an "age" column


class TestCreateModelRegistry:
    def test_builds_registry_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path)
        monkeypatch.setattr(utils, "BASE_DIR", str(tmp_path))

        disease_dir = tmp_path / "diabetes"
        disease_dir.mkdir()
        (disease_dir / "config.json").write_text(json.dumps({"target_column": "outcome"}))
        (disease_dir / "input_fields.json").write_text(
            json.dumps({"disease_name": "Diabetes", "features": ["age", "bmi"]})
        )

        registry = utils.create_model_registry()

        assert "Diabetes" in registry
        assert registry["Diabetes"]["folder_name"] == "diabetes"
        assert registry["Diabetes"]["target_column"] == "outcome"
        assert (tmp_path / "model_registry.json").exists()

    def test_skips_folders_missing_required_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path)
        monkeypatch.setattr(utils, "BASE_DIR", str(tmp_path))
        (tmp_path / "incomplete_model").mkdir()

        registry = utils.create_model_registry()

        assert registry == {}

    def test_returns_empty_registry_when_models_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path / "does_not_exist")
        registry = utils.create_model_registry()
        assert registry == {}


class TestGetAvailableModels:
    def test_marks_model_valid_only_with_all_required_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path)

        complete = tmp_path / "diabetes"
        complete.mkdir()
        (complete / "config.json").write_text("{}")
        (complete / "input_fields.json").write_text(json.dumps({"features": ["age"]}))
        (complete / "predictor.pkl").write_text("stub")

        incomplete = tmp_path / "flu"
        incomplete.mkdir()
        (incomplete / "config.json").write_text("{}")

        models = utils.get_available_models()
        by_name = {m["folder_name"]: m for m in models}

        assert by_name["diabetes"]["valid"] is True
        assert by_name["flu"]["valid"] is False

    def test_returns_empty_list_when_models_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "models_dir", tmp_path / "does_not_exist")
        assert utils.get_available_models() == []
