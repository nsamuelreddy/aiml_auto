# AutoML Studio

AutoML Studio is an end-to-end Python-based AutoML platform that automates the machine learning workflow for tabular datasets.

Users can upload a dataset, select the target column, run an automated machine learning pipeline, compare trained models, inspect model performance and feature importance, and use the saved model for predictions.

## Features

* Upload tabular datasets
* Select the target column for prediction
* Automated machine learning pipeline
* Automated data preprocessing
* Model training and comparison
* Performance overview and model leaderboard
* Best model identification
* Detailed model metrics
* Feature importance and model interpretability
* Save trained models and preprocessing artifacts
* Download generated reports
* Reuse saved models for predictions
* Interactive prediction interface

## How It Works

The application follows an end-to-end machine learning workflow:

```text
Dataset Upload
      |
      v
Target Column Selection
      |
      v
Data Preprocessing
      |
      v
Model Training
      |
      v
Model Comparison
      |
      v
Best Model Selection
      |
      v
Model Evaluation
      |
      v
Model & Preprocessing Artifacts
      |
      v
Prediction
```

The training process saves reports, the final model, and preprocessing artifacts so that the trained pipeline can be reused for prediction.

## Main Components

### Dataset Input

Users provide:

* Dataset file
* Target column

The selected dataset and target column are used to start the AutoML pipeline.

### Model Comparison

After training, the application provides a performance overview and compares the trained models.

The interface includes:

* Best result
* Top model
* Model performance comparison
* Score distribution
* Model leaderboard
* Detailed metrics

### Model Interpretability

The platform provides model insights and feature importance information for models where explainability data is available.

This helps users understand which features contribute most to the model's predictions.

### Prediction

The selected trained model can be reused for predictions.

Users can enter values for the model's selected features and generate predictions using the saved model.

## Key Capabilities

The platform automates several parts of the traditional machine learning workflow, reducing the amount of manual code required to train and evaluate models.

Instead of manually implementing the complete workflow for every dataset, the user can provide a dataset and target column and allow the platform to perform the automated pipeline.

## Technology Stack

* Python
* Machine Learning
* FastAPI
* Scikit-learn
* HTML
* CSS
* Bootstrap
* JavaScript
* Fetch API
* REST API

## Project Architecture

```text
User
 |
 v
Web Interface
 |
 v
FastAPI Backend
 |
 v
AutoML Pipeline
 |
 +---- Data Preprocessing
 |
 +---- Model Training
 |
 +---- Model Comparison
 |
 +---- Model Evaluation
 |
 +---- Feature Importance
 |
 v
Saved Model + Preprocessing Artifacts
 |
 v
Prediction
```

## Live Demo

[AutoML Studio](https://aiml-auto.onrender.com/)

## Project Workflow

1. Upload a tabular dataset.
2. Select the target column.
3. Start the AutoML pipeline.
4. Monitor the training process.
5. Compare model performance.
6. Identify the best-performing model.
7. Review detailed model metrics.
8. Inspect feature importance where available.
9. Download reports and saved artifacts.
10. Reuse the saved model for predictions.

## Why This Project

Traditional machine learning workflows require repeated implementation of preprocessing, model training, evaluation, and prediction steps for every new dataset.

AutoML Studio was developed to automate these repetitive steps and provide a unified interface for experimenting with tabular machine learning workflows.

## Future Improvements

* Support for larger datasets
* Additional machine learning algorithms
* Advanced hyperparameter optimization
* Improved model explainability
* Dataset profiling and automated EDA
* Support for regression workflows
* Model deployment options
* Improved prediction input generation

## Author

**N Samuel Reddy**

B.Tech in Computer Science and Engineering
Rajiv Gandhi University of Knowledge Technologies (RGUKT), RK Valley
