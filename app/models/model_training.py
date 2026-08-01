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


def get_models():

    models = {

        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Random Forest": RandomForestClassifier(random_state=42),

        "KNN": KNeighborsClassifier(),

        "SVM": CalibratedClassifierCV(
            estimator=SVC(),
            ensemble=False
        ),
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

    return models


def train_models(
    x_train,
    y_train,
    x_test,
    progress_callback=None
):

    trained_models = {}

    predictions = {}

    models = get_models()

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

    return trained_models, predictions