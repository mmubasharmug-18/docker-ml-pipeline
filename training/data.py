import hashlib
from pathlib import Path

import pandas as pd

FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]
TARGET = "outcome"
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
MIN_ROWS = 50

RENAMES = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree",
    "Age": "age",
    "Outcome": "outcome",
}


class DataError(ValueError):
    pass


def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Dataset not found: {path}. Put diabetes.csv in the data folder.")

    try:
        frame = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise DataError(f"Could not read {path} as a CSV file: {error}")

    frame = frame.rename(columns=RENAMES)
    missing = [name for name in FEATURES + [TARGET] if name not in frame.columns]
    if missing:
        raise DataError(f"Dataset is missing columns: {', '.join(missing)}")

    frame = frame[FEATURES + [TARGET]].copy()
    for column in frame.columns:
        numbers = pd.to_numeric(frame[column], errors="coerce")
        bad = int(numbers.isna().sum())
        if bad:
            raise DataError(f"Column {column} has {bad} empty or non-numeric value(s)")
        frame[column] = numbers

    if not set(frame[TARGET].unique()) <= {0, 1}:
        raise DataError("Column outcome must only contain 0 and 1")
    if frame[TARGET].nunique() < 2:
        raise DataError("Column outcome needs both 0 and 1 to train a classifier")
    if len(frame) < MIN_ROWS:
        raise DataError(f"Dataset has only {len(frame)} rows, at least {MIN_ROWS} are needed")

    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
