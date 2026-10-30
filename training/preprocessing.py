from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.data import ZERO_AS_MISSING


def build_preprocessor() -> Pipeline:
    fill_zeros = ColumnTransformer(
        [("median_for_zeros", SimpleImputer(missing_values=0, strategy="median"), ZERO_AS_MISSING)],
        remainder="passthrough",
    )
    return Pipeline([("fill_zeros", fill_zeros), ("scale", StandardScaler())])
