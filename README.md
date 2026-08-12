# Machine Learning Assignment 2 — Mobile Price Classification

**Student:** Aman Singh  
**BITS ID:** 2025AC05123  
**Dataset:** Mobile Price Classification  
**GitHub repository:** `REPLACE_WITH_AMAN_GITHUB_REPOSITORY_URL`  
**Live Streamlit app:** `REPLACE_WITH_AMAN_STREAMLIT_APP_URL`

## A. Problem Statement

The objective is to predict the price category of a mobile phone from 20 hardware and connectivity attributes. This is a four-class classification problem. Five classifiers explicitly listed in the assignment are trained on the same fixed train/test split and compared using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The deployed Streamlit application lets the evaluator select a saved model, upload labelled test data, view evaluation metrics and a confusion matrix, or generate predictions for an unlabelled mobile dataset.

## B. Dataset Description

**Public source:** https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

The supplied data contains two CSV files:

- `data/mobile_price_train.csv`: 2,000 labelled records, 20 input features, and the target `price_range`.
- `data/mobile_price_unlabelled.csv`: 1,000 unlabelled records with an `id` column and the same 20 input features.

The target classes are:

| Class | Meaning |
|---:|---|
| 0 | Budget |
| 1 | Lower Mid-Range |
| 2 | Upper Mid-Range |
| 3 | Premium |

The labelled file is perfectly balanced, with 500 records in each class. It contains no missing values and no duplicate rows. The original unlabelled file cannot be used to calculate evaluation metrics because it has no `price_range` column.

### Data preparation

1. The 20 predictor columns are separated from `price_range`.
2. A single stratified 80/20 split is created with random state 123.
3. The 1,600 training rows are used to fit every model.
4. The untouched 400-row test partition is saved as root-level `test_data.csv`.
5. Median imputation is included inside every pipeline as a defensive step.
6. Logistic Regression and Gaussian Naive Bayes use standard scaling.
7. kNN uses robust scaling because it is distance-based and sensitive to feature scale.
8. Decision Tree and Random Forest do not use scaling.
9. Complete preprocessing-plus-model pipelines are saved with Joblib.

For this multiclass problem, Precision, Recall, and F1 use weighted averaging. AUC uses weighted one-vs-rest evaluation.

## C. GitHub Repository Link

`REPLACE_WITH_AMAN_GITHUB_REPOSITORY_URL`

## D. Models Used and Results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9625 | 0.9986 | 0.9624 | 0.9625 | 0.9622 | 0.9502 |
| Decision Tree | 0.8175 | 0.9003 | 0.8214 | 0.8175 | 0.8182 | 0.7574 |
| kNN | 0.6350 | 0.8346 | 0.6510 | 0.6350 | 0.6412 | 0.5142 |
| Naive Bayes | 0.8050 | 0.9453 | 0.8128 | 0.8050 | 0.8075 | 0.7408 |
| Random Forest (Ensemble) | 0.8825 | 0.9777 | 0.8845 | 0.8825 | 0.8826 | 0.8439 |

### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model. It achieved the highest Accuracy, F1, MCC, and AUC. The result indicates that the four price bands are strongly separable using a mostly linear combination of the supplied features, especially high-impact numeric variables such as RAM and battery/display specifications. |
| Decision Tree | Easier to interpret than the ensemble but less accurate. Limiting depth and leaf size reduces unrestricted overfitting, although a single tree still makes less stable boundaries than Logistic Regression or Random Forest. |
| kNN | Lowest-performing model. Robust scaling improves the distance calculation, but all 20 dimensions still make local-neighbour decisions difficult. This is an example of how distance-based models can be affected by higher-dimensional feature spaces. |
| Naive Bayes | Fast and reasonably strong, with AUC above 0.94. Its conditional-independence assumption is restrictive because several mobile specifications are related, so its final class predictions are weaker than Logistic Regression and Random Forest. |
| Random Forest (Ensemble) | Second-best overall. It provides strong nonlinear performance and a high AUC, but remains below Logistic Regression on the fixed test split. It is less directly interpretable and creates a larger saved model. |
| Overall Winner | **Logistic Regression** because it has the strongest combined Accuracy, weighted F1, AUC, and MCC on the untouched test data. |

## Streamlit Application

The UI is intentionally simple. It contains three views:

1. **Compare models** — shows the saved six-metric table and selected-model scores.
2. **Evaluate labelled CSV** — accepts `test_data.csv` or another compatible labelled CSV, then displays metrics, a four-class confusion matrix, a classification report, and downloadable predictions.
3. **Predict unlabelled CSV** — uses the supplied Kaggle test file or another compatible unlabelled CSV and returns the predicted class, readable price label, and class probabilities.

The app never trains on uploaded files. It only loads saved pipelines and uses uploads for evaluation or prediction.

## Repository Structure

```text
aman_mobile_price_assignment/
├── streamlit_app.py
├── README.md
├── requirements.txt
├── test_data.csv
├── data/
│   ├── mobile_price_train.csv
│   └── mobile_price_unlabelled.csv
├── model/
│   ├── mobile_training.py
│   ├── model_card.json
│   ├── comparison.csv
│   └── saved_models/
│       ├── logistic_regression.joblib
│       ├── decision_tree.joblib
│       ├── knn.joblib
│       ├── naive_bayes.joblib
│       └── random_forest.joblib
├── notebook/
│   └── mobile_price_models.ipynb
└── .streamlit/
    └── config.toml
```

## Run Locally on macOS

Open Terminal in this project folder and run:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Rebuild the test split, comparison table, metadata, and all five model files:

```bash
python -m model.mobile_training
```

Launch Jupyter and run the notebook from top to bottom:

```bash
jupyter lab
```

If Jupyter is not installed in the environment, first run:

```bash
python -m pip install jupyterlab
```

Launch the application:

```bash
streamlit run streamlit_app.py
```

Open the local address shown in Terminal, normally `http://localhost:8501`.

## Test the App Before Deployment

1. Open **Compare models** and confirm all five rows are visible.
2. Select each model and verify the six saved scores appear.
3. Open **Evaluate labelled CSV** and use the bundled `test_data.csv`.
4. Confirm metrics, confusion matrix, classification report, and prediction download work.
5. Upload a CSV with one required feature removed and confirm the app shows a readable error.
6. Open **Predict unlabelled CSV** and use the bundled Kaggle test file.
7. Confirm the output contains `id`, predicted class, readable class label, and four probability columns.

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository in Aman's GitHub account.
2. Commit this project from the repository root.
3. Push the `main` branch to GitHub.
4. Open Streamlit Community Cloud and create a new app.
5. Select the repository and branch `main`.
6. Use `streamlit_app.py` as the entrypoint.
7. Select Python 3.13 in advanced settings.
8. Deploy and inspect the build log.
9. Test the public app using the bundled and uploaded CSV options.
10. Replace both placeholder URLs at the top of this README and push the final change.

## Suggested Git Commands

```bash
git init -b main
git add .
git commit -m "Create mobile price classification workflow"
git remote add origin REPLACE_WITH_AMAN_GITHUB_REPOSITORY_URL
git push -u origin main
```

Create later commits only after genuine changes, for example model implementation, UI creation, README completion, and deployment fixes.

## BITS Virtual Lab and Submission PDF

Run the same repository in the BITS Virtual Lab, execute the training script or notebook, and capture one readable screenshot as proof. The final PDF should contain, in the required order:

1. GitHub repository link
2. Live Streamlit application link
3. One BITS Virtual Lab execution screenshot
4. The complete final README content

## Limitations

- The class names are descriptive labels assigned for UI readability; the original target is numeric 0–3.
- The separate Kaggle test file has no ground-truth target, so it supports prediction only.
- Results are specific to the fixed split and selected hyperparameters.
- This project is an educational classification demonstration, not a commercial pricing engine.
