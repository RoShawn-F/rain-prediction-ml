# Rain Prediction in the Melbourne Area 🌧️

A machine learning classification project that predicts whether it will rain today in the Melbourne area, using the previous day's weather observations. It compares a **Random Forest** and a **Logistic Regression** model, both tuned with cross-validated grid search inside a scikit-learn pipeline.

📓 **[View the notebook](rain_prediction_melbourne.ipynb)**

---

## The Problem

Rain is the minority class (about 24% of days), so a "model" that always predicts *no rain* is already 76% accurate while catching zero rainy days. The goal was to beat that baseline **and** actually identify rainy days, which means looking beyond accuracy to recall, precision and the confusion matrix.

## Data

- **Source:** "Rain in Australia" dataset (Kaggle), originally from the Australian Bureau of Meteorology
- **Scope:** about 145,000 daily observations from 49 locations, 2008 to 2017
- **Used for modeling:** 7,557 complete daily records from three nearby stations: Melbourne, Melbourne Airport and Watsonia

## Approach

1. **Cleaning:** dropped rows with missing values (about 56,000 complete rows remained across all locations).
2. **Avoiding data leakage:** several features (rainfall, sunshine, evaporation, max temperature, wind gusts) describe the whole day, so they can't be known in time to forecast that same day. The problem was reframed as *predict today's rain from data up to and including yesterday*, and the columns were renamed to match (`RainYesterday`, `RainToday`).
3. **Regional focus:** narrowed to the Melbourne area so the model learns one local climate instead of 49.
4. **Feature engineering:** created a `Season` feature from the date (Southern Hemisphere seasons).
5. **Preprocessing pipeline:** `StandardScaler` for numeric features and `OneHotEncoder` for categorical ones, combined in a `ColumnTransformer`.
6. **Modeling:** stratified 80/20 train/test split, then `GridSearchCV` with stratified 5-fold cross-validation for each model.
7. **Evaluation:** classification report, confusion matrix, true positive rate, and feature importance.

## Results

| Model | Accuracy | True positive rate (rain) | Precision (rain) | False alarms |
|---|---|---|---|---|
| Always predict "No rain" (baseline) | 76% | 0% | n/a | 0 |
| Logistic Regression | 83% | 51% (182 / 358) | 68% | 84 |
| **Random Forest** | **84%** | **51% (184 / 358)** | **75%** | **62** |

**Key findings**

- Both models beat the baseline, but only by 7 to 8 points, and both catch about **half** of rainy days. The class imbalance pushes the models toward predicting "no rain."
- The **Random Forest** performed slightly better overall, mainly by producing fewer false alarms.
- **Afternoon humidity (`Humidity3pm`)** was by far the most important feature, followed by afternoon and morning pressure and hours of sunshine. Wind direction contributed very little.

## What I'd Try Next

- Optimize the grid search for **recall or F1** instead of accuracy
- Counter the imbalance with `class_weight='balanced'` or resampling
- Tune the decision threshold to catch more rainy days
- Impute missing values instead of dropping rows, and add more stations or years
- Add lag features, such as the 24-hour change in pressure or humidity

## Tools

Python · pandas · scikit-learn (Pipeline, ColumnTransformer, GridSearchCV, StratifiedKFold, RandomForestClassifier, LogisticRegression) · Matplotlib · Seaborn · Jupyter

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook rain_prediction_melbourne.ipynb
```

The notebook downloads the dataset automatically, so no separate data file is needed.

---

*Completed as the final project for the IBM **Machine Learning with Python** course, part of the IBM Data Science Professional Certificate on Coursera.*
