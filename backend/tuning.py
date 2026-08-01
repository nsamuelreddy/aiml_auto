from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
def tune_model(
        model,
        param_space,
        x_train,
        y_train,
        search_type="grid",
        cv=3,
        scoring='accuracy',
        n_iter=5,
        random_state=42,
        n_jobs=2
):
    if search_type == "grid":
        search=GridSearchCV(
            estimator=model,
            param_grid=param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs

        )
    elif search_type =="random":
        search=RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("Search type must be grid or random")
    search.fit(x_train,y_train)
    return {
        "best_model":search.best_estimator_,
        "best_params":search.best_params_,
        "best_score":search.best_score_,
        "search":search,
        "cv_results":search.cv_results_
    }