import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Skip SVM when training rows exceed this threshold.
# SVM time complexity is O(n²)–O(n³), making it impractically slow on large data.
SVM_ROW_LIMIT = 10_000


def get_models(skip_svm: bool = False):

    models = {

        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Random Forest": RandomForestClassifier(random_state=42),

        "KNN": KNeighborsClassifier(),

        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        ),

        "LightGBM": LGBMClassifier(
            random_state=42,
            verbose=-1
        )

    }

    if not skip_svm:
        models["SVM"] = CalibratedClassifierCV(
            estimator=SVC(),
            ensemble=False
        )

    return models


def train_models(
    x_train,
    y_train,
    x_test,
    progress_callback=None,
    skip_svm_by_size: bool = False,
):

    trained_models = {}

    predictions = {}

    n_rows = len(x_train)
    skip_svm = skip_svm_by_size or (n_rows > SVM_ROW_LIMIT)

    models = get_models(skip_svm=skip_svm)

    total_models = len(models)

    for index, (name, model) in enumerate(models.items(), start=1):

        if progress_callback is not None:
            progress_callback(index - 1, total_models, name)

        model.fit(x_train, y_train)

        prediction = model.predict(x_test)

        trained_models[name] = model

        predictions[name] = prediction

        if progress_callback is not None:
            progress_callback(index, total_models, name)

    if skip_svm and progress_callback is not None:
        if skip_svm_by_size:
            reason = "file size > 1.5 MB"
        else:
            reason = f"{n_rows:,} rows > {SVM_ROW_LIMIT:,} limit"
        progress_callback(total_models, total_models, f"SVM skipped ({reason})")

    return trained_models, predictions