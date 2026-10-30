from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def candidate_models(random_state: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
