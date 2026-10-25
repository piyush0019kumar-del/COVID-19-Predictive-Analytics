# COVID-19-Predictive-Analytics
Predictive analytics project comparing six machine learning classification algorithms on COVID-19 survey data.
# COVID-19 Predictive Analytics

## Project Overview
This project applies machine learning classification algorithms to analyze survey data collected during the COVID-19 lockdown. The objective is to predict whether respondents felt their time during lockdown was well spent (`YES` or `NO`).

This project was developed as part of my B.Tech Data Science learning journey to gain practical experience in predictive analytics and machine learning.

## Problem Statement
To build and compare multiple classification models using survey responses related to online classes, self-study, sleep, physical activity, social media usage, and other lockdown experiences.

## Dataset
- **Records:** 1,182
- **Original columns:** 27
- **Target variable:** `time_well.spent`
- **Prediction type:** Binary classification (`YES` / `NO`)

The dataset includes numerical and categorical information about respondents' lockdown activities.

## Machine Learning Algorithms
Six classification algorithms were implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)
6. Random Forest

## Data Preprocessing
- Checked missing values
- Imputed missing numerical and categorical values
- Encoded categorical features using One-Hot Encoding
- Scaled numerical features
- Split the data into 80% training and 20% testing sets
- Excluded identifier columns and selected potentially leakage-prone features

## Evaluation Metrics
The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Results

Results from the current train-test split:

| Algorithm | Accuracy | F1-score |
|---|---:|---:|
| Random Forest | 63.71% | 61.26% |
| Support Vector Machine | 61.60% | 62.24% |
| Logistic Regression | 61.60% | 61.28% |
| Decision Tree | 61.18% | 61.67% |
| K-Nearest Neighbors | 54.43% | 53.45% |
| Gaussian Naive Bayes | 51.48% | 10.85% |

**Key findings:**
- Random Forest achieved the highest accuracy of 63.71%.
- Support Vector Machine achieved the highest F1-score of 62.24%.
- The results indicate modest predictive performance and scope for further improvement.

These results are from one train-test split and should not be interpreted as a guarantee of performance on unseen populations.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- OpenPyXL

## How to Run

1. Install the required packages:

   `pip install pandas openpyxl scikit-learn matplotlib`

2. Place the Excel dataset in the same folder as the Python script. Obtain it from an authorized source; it is not included in this repository.

3. Run the Python script using IDLE or execute:

   `python "Impact of Social Media on Student Life.py"`

4. Review the evaluation metrics, confusion matrices, comparison table, and performance graph.

## Future Improvements
- Apply cross-validation
- Perform systematic feature selection
- Investigate remaining target leakage
- Tune model hyperparameters
- Compare additional regression, clustering, and neural network approaches

## Author
B.Tech Computer Science and Engineering | Data Science

## Project Purpose
An educational project demonstrating data preprocessing, supervised machine learning, classification evaluation, and model comparison.
