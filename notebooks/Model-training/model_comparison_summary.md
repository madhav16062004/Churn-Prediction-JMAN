# Model Comparison Summary

This summary is based on the recorded outputs currently saved inside the model notebooks in [notebooks/Model-training](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training).

## Best Recorded Result Per Notebook

| Model | Notebook | Selected variant | Accuracy | Precision | Recall | F1 | ROC AUC | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| XGBoost | [05_model_XGBoost.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/05_model_XGBoost.ipynb) | Tuned | 0.9565 | 0.8024 | 0.8074 | 0.8049 | 0.9806 | Best overall F1 among recorded runs |
| LightGBM | [04_Model_GBM.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/04_Model_GBM.ipynb) | Tuned | 0.9549 | 0.7895 | 0.8110 | 0.8001 | 0.9808 | Best ROC AUC among recorded runs |
| Gradient Boosting | [02_model_gradientboost.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/02_model_gradientboost.ipynb) | Tuned | 0.9532 | 0.7754 | 0.8149 | 0.7947 | 0.9777 | Strong balanced performer |
| AdaBoost | [01_model_Adaboost.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/01_model_Adaboost.ipynb) | Tuned | 0.9353 | 0.6640 | 0.8477 | 0.7447 | 0.9711 | Highest recall among tuned boosting models |
| Decision Tree | [09_model_decisiontree.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/09_model_decisiontree.ipynb) | With SMOTE | 0.9432 | 0.7106 | 0.8256 | 0.7638 | N/A | Better recall and slightly better F1 than no-SMOTE tree |
| Random Forest | [10_model_randomforest.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/10_model_randomforest.ipynb) | Without SMOTE | 0.9421 | 0.9037 | 0.5371 | 0.6738 | N/A | Highest precision, but low recall |
| Logistic Regression | [03_model_Logistic.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/03_model_Logistic.ipynb) | Baseline | 0.8909 | 0.5056 | 0.8571 | 0.6360 | 0.9529 | Very recall-heavy, weak precision |
| KNN | [08_model_knn.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/08_model_knn.ipynb) | Tuned | 0.9187 | 0.6740 | 0.5221 | 0.5884 | 0.8845 | Middling precision, limited recall |
| Naive Bayes | [06_model_naive.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/06_model_naive.ipynb) | Baseline | 0.8743 | 0.4595 | 0.7356 | 0.5656 | 0.9028 | Weakest overall among current runs |
| SVM | [07_model_svm.ipynb](C:/Users/madha/OneDrive/Desktop/DS/Churn-Prediction-JMAN/notebooks/Model-training/07_model_svm.ipynb) | Tuned | 0.8508 | 0.4132 | 0.8118 | 0.5477 | 0.9157 | Recall-heavy, but low precision and F1 |

## Ranking By F1

| Rank | Model | Selected variant | F1 | Precision | Recall | Accuracy |
|---:|---|---|---:|---:|---:|---:|
| 1 | XGBoost | Tuned | 0.8049 | 0.8024 | 0.8074 | 0.9565 |
| 2 | LightGBM | Tuned | 0.8001 | 0.7895 | 0.8110 | 0.9549 |
| 3 | Gradient Boosting | Tuned | 0.7947 | 0.7754 | 0.8149 | 0.9532 |
| 4 | Decision Tree | With SMOTE | 0.7638 | 0.7106 | 0.8256 | 0.9432 |
| 5 | AdaBoost | Tuned | 0.7447 | 0.6640 | 0.8477 | 0.9353 |
| 6 | Random Forest | Without SMOTE | 0.6738 | 0.9037 | 0.5371 | 0.9421 |
| 7 | Logistic Regression | Baseline | 0.6360 | 0.5056 | 0.8571 | 0.8909 |
| 8 | KNN | Tuned | 0.5884 | 0.6740 | 0.5221 | 0.9187 |
| 9 | Naive Bayes | Baseline | 0.5656 | 0.4595 | 0.7356 | 0.8743 |
| 10 | SVM | Tuned | 0.5477 | 0.4132 | 0.8118 | 0.8508 |

## Tradeoff View

| Goal | Best current choice | Why |
|---|---|---|
| Best overall churn-detection balance | XGBoost | Highest recorded F1 with strong precision and recall |
| Best ranking/separation ability | LightGBM | Highest ROC AUC |
| Highest recall | Logistic Regression | Recall 0.8571, but low precision |
| Highest precision | Random Forest without SMOTE | Precision 0.9037, but recall drops to 0.5371 |
| Best simple tree option | Decision Tree with SMOTE | Better recall/F1 than no-SMOTE tree |
| Best classical non-ensemble baseline | Logistic Regression | Better F1 than SVM, KNN, and Naive Bayes |

## Notebook-Level Observations

- The top tier is clear: XGBoost, LightGBM, and Gradient Boosting are clustered well above the other models on F1 and accuracy.
- AdaBoost is still competitive if recall matters more than precision.
- Logistic Regression is useful as a high-recall baseline, but it over-flags churn compared with the top tree/boosting models.
- Decision Tree with SMOTE is a decent interpretable option, though still behind the boosted models.
- Random Forest shows the classic tradeoff here: excellent precision without SMOTE, but it misses many churn cases.
- KNN, Naive Bayes, and SVM are materially weaker on the saved outputs.

## Recommendation

If you want one model to carry forward, use XGBoost as the primary candidate and LightGBM as the closest backup.

If your business goal prioritizes catching as many churn cases as possible, keep Logistic Regression or AdaBoost as recall-oriented reference models when comparing operating thresholds.
