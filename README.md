# Dockerized ML Training and Deployment Pipeline

Two containers that share a folder of saved models.

- **Trainer**: loads the diabetes dataset, cleans it, trains three models, compares them, keeps the best one and saves it as a numbered version.
- **API**: a separate FastAPI service that loads the saved model and answers `POST /predict`. It checks every input and tells you which model version answered.

```
data/diabetes.csv  ->  [trainer container]  ->  models volume (v1, v2, ...)  ->  [api container]  ->  /predict
```

This is a learning project, not medical advice.

## Folder structure

```
diabetes_ml_pipeline/
  training/
    config.py          settings from environment variables
    data.py            load and check the CSV
    preprocessing.py   fills zeros, scales the numbers
    models.py          the three models
    evaluate.py        metrics and choosing the best model
    registry.py        version numbers, saving, the "current" pointer
    pipeline.py        runs everything in order
    __main__.py        python -m training
  serving/
    config.py          settings from environment variables
    schemas.py         input and output models (the validation rules)
    errors.py          readable error messages
    handlers.py        validation errors return 400
    model_store.py     finds and loads a model version
    main.py            the FastAPI app
    healthcheck.py     used by Docker
    __main__.py        python -m serving
  tests/
    training/          tests for the trainer
    serving/           tests for the API
  samples/             JSON bodies for curl
  data/                put diabetes.csv here
  models/              saved models when running without Docker
  Dockerfile.train
  Dockerfile.api
  docker-compose.yml
  requirements-ml.txt  pinned ML libraries, shared by both images
  requirements-api.txt
  requirements-dev.txt
  pytest.ini
```

## Dataset

Pima Indians Diabetes dataset from Kaggle, the same `diabetes.csv` as before. Put it in the `data` folder. It has 768 rows, 8 measurements and an `Outcome` column (1 means diabetes).

In five columns (glucose, blood pressure, skin thickness, insulin, BMI) a zero really means "not measured". The preprocessing treats those zeros as missing.

## What the trainer does

1. Loads the CSV and checks it: all columns present, only numbers, outcome is 0 or 1, no empty cells.
2. Splits it 80% train and 20% test, keeping the same share of diabetic patients in both.
3. Preprocessing: replace the "not measured" zeros with the median of that column, then scale every column.
4. Trains three models: Logistic Regression, Random Forest and Gradient Boosting.
5. Scores each model with 5-fold cross-validation on the training rows. The preprocessing is inside every fold, so nothing leaks.
6. Picks the best model by one metric. The default is `roc_auc`. Change it with `SELECTION_METRIC`.
7. Trains all three on the training rows and scores them once on the test rows, which no model has seen.
8. Saves the winner, the fitted preprocessing and a metadata file as a new version.

The choice is made on the cross-validation scores, not the test scores. That keeps the test set honest as the final check.

Metrics:

| Metric | Meaning |
| --- | --- |
| precision | Of the patients the model called diabetic, how many really were |
| recall | Of the patients who really are diabetic, how many the model found |
| f1 | One number that balances precision and recall |
| roc_auc | How well the model ranks diabetic patients above healthy ones (1.0 is perfect, 0.5 is guessing) |
| accuracy | Share of all predictions that were right |

For a health problem, recall matters because a missed patient costs more than a false alarm. You can select on it with `SELECTION_METRIC=recall`.

## What gets saved

Inside the models folder:

```
models/
  current.json          {"version": "v2"}
  v1/
    model.joblib         the trained model
    preprocessor.joblib  the fitted preprocessing
    metadata.json
  v2/
    ...
```

`metadata.json` records the version, time, model name, the metric used and its score, the scores of all three models, the confusion matrix, the 8 feature names, the dataset size and a hash of the file, the split settings and the library versions.

## Model version tracking

- Every training run makes a new folder: `v1`, `v2`, `v3`. Old versions are never overwritten.
- `current.json` points to the newest one. The API uses it unless told otherwise.
- Every `/predict` answer contains `model_version` and `model_name`. `GET /model` shows the scores and all available versions.
- To go back, set `MODEL_VERSION=v1` on the API. It then ignores `current.json`.
- Pickled models only load safely with the same scikit-learn version. The metadata stores the version used for training, and `/model` shows `sklearn_matches`. Both images install the same pinned versions from `requirements-ml.txt`.

## The API

| Method | Path | What it does |
| --- | --- | --- |
| POST | `/predict` | Takes 8 measurements, returns the prediction |
| GET | `/model` | Version, scores and features of the model being served |
| GET | `/health` | 200 when a model is loaded, 503 when not |

Docs page: http://localhost:8000/docs

Request:

```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 25,
  "insulin": 100,
  "bmi": 32.5,
  "diabetes_pedigree": 0.45,
  "age": 35
}
```

Answer (your numbers will differ):

```json
{
  "prediction": 0,
  "label": "not diabetic",
  "probability": 0.21,
  "threshold": 0.5,
  "model_version": "v1",
  "model_name": "Logistic Regression"
}
```

Validation rules. A request that breaks any rule gets **400** and nothing is predicted. Every problem is listed.

| Field | Rule |
| --- | --- |
| pregnancies | whole number, 0 to 20 |
| glucose | number, 0 to 300 |
| blood_pressure | number, 0 to 200 |
| skin_thickness | number, 0 to 100 |
| insulin | number, 0 to 900 |
| bmi | number, 0 to 70 |
| diabetes_pedigree | number, 0 to 3 |
| age | whole number, 1 to 120 |

Also refused: missing fields, unknown fields, text, true/false, null, NaN and a body that is not a JSON object. For the five "not measured" columns, 0 is allowed and means missing.

Error example:

```json
{
  "detail": [
    { "field": "glucose", "message": "glucose: Input should be less than or equal to 300 (got 500)" },
    { "field": "age", "message": "age: Input should be less than or equal to 120 (got 150)" }
  ]
}
```

If no model has been trained yet, the API still starts, but `/predict`, `/model` and `/health` return 503 with the message "No trained model found".

## Settings (environment variables)

Trainer:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_PATH` | `data/diabetes.csv` (`/data/diabetes.csv` in Docker) | CSV file |
| `MODEL_DIR` | `models` (`/models` in Docker) | Where versions are saved |
| `SELECTION_METRIC` | `roc_auc` | `roc_auc`, `f1`, `recall`, `precision` or `accuracy` |
| `TEST_SIZE` | `0.2` | Share of rows held back for testing |
| `CV_FOLDS` | `5` | Number of cross-validation folds |
| `RANDOM_STATE` | `42` | Makes the run repeatable |

API:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_DIR` | `models` (`/models` in Docker) | Where to look for models |
| `MODEL_VERSION` | empty | Empty means the current version, `v1` pins that version |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `APP_HOST` | `127.0.0.1` (`0.0.0.0` in Docker) | Address to listen on |
| `APP_PORT` | `8000` | Port inside the container |

A wrong value stops the program straight away with a clear message and exit code 2.

## Run with Docker

Docker Desktop must be running. Put `diabetes.csv` in the `data` folder first, then open a terminal in the project folder (PowerShell).

**1. Train.** This builds the trainer image the first time and saves `v1` into the shared volume `diabetes-models`:

```
docker compose run --rm trainer
```

It prints both score tables and the selected model.

**2. Start the API:**

```
docker compose up -d --build api
docker compose ps
```

After about 30 seconds the status says `(healthy)`.

**3. Call it from my computer** (use `curl.exe` in PowerShell):

```
curl.exe http://localhost:8000/health

curl.exe http://localhost:8000/model

curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/valid.json

curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/high_risk.json

curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/not_measured.json
```

Error cases (`-i` shows the status line):

```
curl.exe -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/bad_range.json

curl.exe -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/bad_types.json

curl.exe -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/missing_field.json
```

Or open http://localhost:8000/docs, choose **POST /predict**, click **Try it out** and **Execute**.

**4. Train again and watch the version change:**

```
docker compose run --rm -e SELECTION_METRIC=recall trainer
docker compose restart api
curl.exe http://localhost:8000/model
```

The new run is `v2`, selected by recall. After the restart the API serves `v2`. Go back to `v1`:

```
$env:MODEL_VERSION="v1"
docker compose up -d api
curl.exe http://localhost:8000/model
```

`pinned_version` is `v1` and `current_version` is still `v2`.

**5. See the logs and stop everything:**

```
docker compose logs api
docker compose down
```

The models stay in the volume `diabetes-models`. `docker volume rm diabetes-models` deletes them.

### Without compose

```
docker build -f Dockerfile.train -t diabetes-trainer .
docker build -f Dockerfile.api -t diabetes-api .
docker volume create diabetes-models
docker run --rm -v "${PWD}/data:/data:ro" -v diabetes-models:/models diabetes-trainer
docker run -d --name diabetes-api -p 8000:8000 -v diabetes-models:/models:ro diabetes-api
```

Why it is built this way:

- The data folder is mounted read-only (a bind mount). The data never goes into an image.
- The models volume is written by the trainer and mounted read-only in the API. The API can read models but cannot change them.
- The two images share no code. They only agree on the files in the volume and on the 8 feature names, which the API checks when it loads a model.
- Both images run as a normal user, not root.

## Run without Docker (VS Code)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
python -m training
python -m serving
```

The first command trains from `data\diabetes.csv` into `models\`. The second one serves it on http://127.0.0.1:8000. The Run and Debug panel has "Train models" and "Run API" too.

## Tests

```
pytest
```

About 170 tests, in two groups.

Trainer:

- settings, bad values reported together
- data loading: renamed columns, missing file, missing columns, text, empty cells, bad outcome values, too few rows
- preprocessing: zeros become the median, real zeros (pregnancies) are kept, output is scaled
- metrics against hand calculations, best-model choice, ties, the metric you ask for
- version numbers (v10 comes after v9), safe saving, the `current.json` pointer
- a full run: three models compared on all five metrics, the saved model reproduces the reported test scores, a second run makes `v2`, the same settings give the same scores, bad data saves nothing
- the `python -m training` command and its exit codes

API:

- a valid prediction, the answer is consistent with the threshold, a high-risk patient scores higher, same input gives the same output
- every field on and just outside its limits, wrong types, missing and unknown fields, NaN, broken JSON, all problems listed together
- no model gives 503, a pinned version, a missing version, a model with the wrong features, a different scikit-learn version
- the real server started on a free port, plus the health check

The tests train their own small model on made-up data. They do not need `diabetes.csv`.

## My results

### 1. Which three models did the trainer compare? Which was selected, by which metric, and with what score?

The trainer compared three machine learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting

**Selected Model:** Logistic Regression  
**Selection Metric:** ROC-AUC  
**Cross-Validation Score:** 0.8434

Logistic Regression was selected because it achieved the highest mean ROC-AUC score across 5-fold cross-validation.

### 2. What were the precision, recall, F1 and ROC-AUC of the selected model on the test set?

The selected Logistic Regression model achieved the following test-set metrics:

| Metric | Score |
|---|---:|
| Precision | 0.6000 |
| Recall | 0.5000 |
| F1-Score | 0.5455 |
| ROC-AUC | 0.8130 |

The model achieved 70.78% test accuracy.

### 3. What did `/predict` return for `valid.json` and `high_risk.json`? Which `model_version` was in the answer?

Both prediction requests were executed successfully.

- `valid.json`: Returned a successful prediction with class 0 (not diabetic).
- `high_risk.json`: Returned a successful prediction with class 1 (diabetic).

Both responses used model version `v1` (Logistic Regression).

### 4. What happened for `bad_range.json`, `bad_types.json` and `missing_field.json`?

The API rejected invalid input requests with validation errors.

- `bad_range.json`: Rejected because one or more feature values were outside the permitted range.
- `bad_types.json`: Rejected because one or more feature values had invalid data types.
- `missing_field.json`: Rejected because one or more required input fields were missing.

The API handled these invalid requests without crashing.

### 5. What changed after training again with `SELECTION_METRIC=recall`? What did `/model` show, and how did I go back to `v1`?

I retrained the models using:

```powershell
docker compose run --rm -e SELECTION_METRIC=recall trainer
```

## Conclusion

- Training and serving are two separate programs. They only meet in the saved files.
- Choose the model on cross-validation, then check it once on data it has not seen.
- Save the preprocessing together with the model, or the API would feed the model different numbers than it was trained on.
- A version number, a metadata file and a pinned library list make it possible to explain and undo what the API is serving.
- Check input at the door. The model never sees a bad request.

## Limits

- The model is trained on 768 rows and is not a medical tool. Do not use real patient data.
- The threshold is a fixed 0.5. A lower one would find more patients and give more false alarms.
- Models are chosen from only three options with default settings. There is no tuning.
- No login. Anyone who can reach the port can call the API.
- The API loads the model once when it starts. After training again, restart it.
- Pinned library versions can disappear from PyPI or stop supporting a new Python. If `pip` cannot install them, change the numbers in `requirements-ml.txt`, then retrain so both images agree again.
