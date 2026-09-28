# AutoML Studio

AutoML Studio is a web application for uploading tabular datasets, training multiple machine learning models automatically, comparing results, and making predictions from the best trained model.

## What the website does

The application is built around a simple flow:

1. Upload a dataset or load the built-in demo dataset.
2. Select the target column.
3. Run the AutoML pipeline.
4. Review results, preprocessing details, model comparison, and feature importance.
5. Use the prediction form to test the trained model.
6. Download the best trained model as a `.pkl` file.

## Main user interface

The website has four main navigation areas:

- Upload: upload a file, choose the target column, and start training.
- Results: view dataset preview, model comparison, EDA charts, and leaderboard data.
- Predict: enter feature values and get a prediction from the selected best model.
- Details: inspect preprocessing steps, model tuning notes, and feature importance.

The interface also includes a light/dark theme toggle. The default theme is light unless the user has already saved a preference in the browser.

## Supported dataset input

The app accepts:

- `.csv`
- `.xlsx`
- `.xls`
- `.json`

Upload size is limited to 20 MB in the browser flow.

There is also a demo dataset loader for the Titanic survival dataset.

## Processing pipeline

After upload, the backend runs a full pipeline that includes:

- dataset loading and preview generation
- cleaning column names
- removing empty rows and columns
- removing duplicate rows
- dropping irrelevant columns
- filling missing values
- encoding categorical features
- scaling numerical features
- feature selection
- train/test split
- model training
- hyperparameter tuning
- model evaluation
- feature importance extraction

The pipeline automatically detects whether the problem is classification or regression.

## Models and metrics

The application trains and compares multiple models, then selects the best one.

For classification, the app reports metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

For regression, the app reports metrics such as:

- R2
- RMSE
- MAE
- MSE

The results page shows a leaderboard and a top-3 comparison view.

## Prediction workflow

After training finishes, the site builds a prediction form from the selected features. The form adapts to the detected feature type:

- dropdowns for categorical or encoded fields
- toggle-style inputs for binary fields
- numeric inputs for continuous values

Prediction is only available after a successful training run, because the app uses the best trained model stored in memory.

## Downloaded model

The site provides a download button for exporting the best model as a serialized Python pickle file. This is useful for local inference or reuse in another Python project.

## Backend behavior

The backend stores uploaded files in the `uploads/` directory and keeps the latest trained model in memory. A small background cleanup task removes old uploaded CSV files automatically.

The main application entry point is `main.py`, and the training pipeline lives in `backend/main.py`.

## API endpoints

- `GET /` renders the main website.
- `POST /api/upload` uploads a dataset and returns column names.
- `GET /api/load-demo` loads the sample Titanic dataset.
- `GET /api/train-stream` runs the AutoML pipeline and streams live progress updates.
- `POST /api/predict` returns a prediction from the best trained model.
- `GET /api/download-model` downloads the best trained model as a `.pkl` file.

## Local run

Install dependencies and start the app:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

If you prefer the project virtual environment:

```bash
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

## Deployment notes

The app is already structured as a standard Python web service, so it can be deployed on platforms such as Render by pointing the service at `main:app` and installing from `requirements.txt`.

## Project structure

- `main.py`: FastAPI application and web UI
- `backend/main.py`: preprocessing, training, tuning, and evaluation pipeline
- `app/`: data preparation, model training, evaluation, comparison, and tuning modules
- `datasets/`: dataset-related resources
- `demo_datasets/`: sample dataset files
- `uploads/`: uploaded files created at runtime
- `saved_models/`: saved model artifacts

## Typical usage

1. Open the site.
2. Upload a dataset or load the demo.
3. Select the target column.
4. Click Run AutoML Pipeline.
5. Review the preview, metrics, model comparison, and preprocessing report.
6. Open Predict and enter feature values.
7. Download the best model if you want to use it elsewhere.

## Notes

- The prediction screen becomes available only after training completes.
- The app uses live progress updates while training is running.
- Large datasets may be sampled down during training to keep the pipeline responsive.
