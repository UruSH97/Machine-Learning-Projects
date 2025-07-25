# 🎬 Movie Review Sentiment Classification (NLP + ML + Feature Engineering)

This project tackles the binary sentiment classification of movie reviews using a combination of **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. It explores multiple classifiers and integrates additional metadata like **box office**, **genre**, **ratings**, and **duration** to improve performance.

---

## 📌 Objective

Predict whether a given movie review expresses **positive** or **negative** sentiment using:
- Textual data (`review_text`)
- Movie metadata (`boxOffice`, `genre`, `audienceScore`, etc.)

---

## 🧰 Tech Stack

- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**
- **TF-IDF Vectorization**
- **Feature Engineering & Stacking**
- **Pipelines, Cross-validation, GridSearchCV**

---

## 📊 Dataset Overview

- **`train.csv`** — contains movie reviews and their associated sentiment (POSITIVE/NEGATIVE).
- **`test.csv`** — similar to train, but without sentiment (for prediction).
- **`movies.csv`** — metadata for each movie (title, description, genre, box office, etc.)

---

## 🔍 Data Pipeline

1. **EDA & Cleaning**:
   - Visualized sentiment distribution, ratings, runtime, box office.
   - Filled missing values using mean/median/mode for numerical/categorical columns.

2. **NLP Preprocessing**:
   - Lowercasing, removing punctuation and HTML tags.
   - Custom stemmer (for 'ing', 'ed', 'ly', etc.)
   - Used `TF-IDF Vectorizer` with custom tokenizer and preprocessor.

3. **Feature Engineering**:
   - Combined structured features (`audienceScore`, `runtimeMinutes`, `boxOffice`, `genre`) with TF-IDF matrix using `hstack`.
   - Normalization using `StandardScaler`.

4. **Modeling**:
   - Baseline Logistic Regression
   - Tuned `LinearSVC` (Best F1: **0.8714**, Test Score: **0.81748**)
   - Tried additional classifiers: `Naive Bayes`, `Decision Trees`, `KNN`, `MLP`, `Bagging`, `XGBoost`, `VotingClassifier`.

---

## 🏆 Key Results

| Model           | F1 Score (Train) | F1 Score (Test) |
|----------------|------------------|-----------------|
| LogisticRegression | 0.8655        | 0.8159          |
| LinearSVC (C=0.334) | **0.8715**   | **0.81748**     |
| With Genre + Box Office | -       | **0.81759**     |

✅ **Adding structured metadata improved the test score** without overfitting.

---

## 📈 Visualizations

- WordClouds
- Rating vs Sentiment Crosstabs
- Correlation heatmaps
- Audience Score & Runtime scatter plots
- Box office distribution histograms

## Skills Demonstrated

- Text preprocessing and feature engineering
- Model stacking and hyperparameter tuning
- Combining numeric and sparse (TF-IDF) features
- GridSearchCV & cross-validation
- Model evaluation using F1, confusion matrix
