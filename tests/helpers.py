import numpy as np
import pandas as pd

VALID_PAYLOAD = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 32.5,
    "diabetes_pedigree": 0.45,
    "age": 35,
}

HIGH_RISK_PAYLOAD = {
    "pregnancies": 6,
    "glucose": 185,
    "blood_pressure": 82,
    "skin_thickness": 35,
    "insulin": 240,
    "bmi": 42.0,
    "diabetes_pedigree": 1.2,
    "age": 58,
}


def make_dataset(rows: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    outcome = (rng.random(rows) < 0.35).astype(int)

    glucose = rng.normal(108, 18, rows) + outcome * 32
    bmi = rng.normal(30, 5, rows) + outcome * 4
    age = rng.integers(21, 65, rows) + outcome * 6
    pregnancies = rng.integers(0, 10, rows)
    blood_pressure = rng.normal(70, 10, rows) + outcome * 4
    skin_thickness = rng.normal(26, 8, rows)
    insulin = rng.normal(110, 40, rows) + outcome * 50
    pedigree = rng.gamma(2.0, 0.25, rows) + outcome * 0.1

    def zero_out(values, share):
        values = values.copy()
        values[rng.random(rows) < share] = 0
        return values

    return pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": zero_out(np.round(glucose), 0.03).astype(int),
            "BloodPressure": zero_out(np.round(blood_pressure), 0.05).astype(int),
            "SkinThickness": zero_out(np.round(skin_thickness), 0.25).astype(int),
            "Insulin": zero_out(np.round(insulin), 0.4).astype(int),
            "BMI": zero_out(np.round(bmi, 1), 0.02),
            "DiabetesPedigreeFunction": np.round(pedigree, 3),
            "Age": age,
            "Outcome": outcome,
        }
    )


def train_into(model_dir, csv_path, **changes) -> dict:
    from training.config import TrainSettings
    from training.pipeline import run

    settings = TrainSettings(data_path=csv_path, model_dir=model_dir, cv_folds=3, **changes)
    return run(settings)
