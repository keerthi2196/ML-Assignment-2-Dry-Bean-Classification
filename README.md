# Dry Bean Variety Classification

## a. Problem statement
The objective is to classify a dry bean sample into one of seven varieties using 16 numerical geometric and shape-related measurements extracted from bean images. Five supervised classification algorithms are trained and compared using the same stratified train/test split.

## b. Dataset description
- **Dataset:** Dry Bean
- **Source:** UCI Machine Learning Repository (Dataset ID: 602)
- **Samples:** 13,611
- **Input features:** 16
- **Target:** `Class`
- **Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
- **Missing values:** The UCI dataset reports no missing values.
- **Dataset citation:** Koklu, M. and Ozkan, I.A. (2020), *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques*, Computers and Electronics in Agriculture. UCI DOI: https://doi.org/10.24432/C50S4B

## c. GitHub Repository Link
Add your repository URL here: **[PASTE GITHUB REPOSITORY URL]**

## d. Models used and evaluation metrics
An 80:20 stratified train/test split was used with `random_state=42`. Precision, recall, and F1 are weighted averages across the seven classes. AUC is calculated using the one-vs-rest (OvR) multiclass method with weighted averaging.

| ML Model Name | Accuracy | AUC (OvR) | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| K-Nearest Neighbors | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Decision Tree | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Support Vector Machine | Accuracy 0.9221, weighted F1 0.9222, AUC 0.9937, and MCC 0.9059. |
| Logistic Regression | Accuracy 0.9207, weighted F1 0.9209, AUC 0.9934, and MCC 0.9042. |
| K-Nearest Neighbors | Accuracy 0.9166, weighted F1 0.9168, AUC 0.9812, and MCC 0.8992. |
| Gaussian Naive Bayes | Accuracy 0.8979, weighted F1 0.8981, AUC 0.9902, and MCC 0.8773. |
| Decision Tree | Accuracy 0.8920, weighted F1 0.8916, AUC 0.9334, and MCC 0.8696. |
| **Overall winner (highest accuracy)** | **Support Vector Machine** with accuracy 0.9221. Consider AUC, F1 and MCC as supporting measures when interpreting the result. |

## Implementation details
- Logistic Regression: StandardScaler + LogisticRegression.
- Decision Tree: DecisionTreeClassifier.
- KNN: StandardScaler + KNeighborsClassifier (k=5).
- Naive Bayes: StandardScaler + GaussianNB.
- SVM: StandardScaler + RBF-kernel SVC with probability estimates enabled.
- Scaling is fitted inside each pipeline using training data only, preventing test-data leakage.

## Streamlit application
Live app URL: **[PASTE STREAMLIT COMMUNITY CLOUD URL]**

The app accepts a labelled test CSV, lets the evaluator select one of the five saved models, and displays accuracy, AUC, precision, recall, F1, MCC, a confusion matrix, and a classification report. The app does not retrain models on uploaded test data.

## How to run locally
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Upload `test_data.csv` from this repository to evaluate the saved models in the app.
