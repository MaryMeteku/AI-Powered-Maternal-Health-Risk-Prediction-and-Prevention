# AI-Powered-Maternal-Health-Risk-Prediction-and-Prevention
#  Overview
This project focuses on developing an AI-powered system to predict and prevent pre-eclampsia in women aged 15–45. Pre-eclampsia is a serious pregnancy complication characterized by high blood pressure and potential damage to organs, most often the liver and kidneys. Early identification is crucial to improving maternal and neonatal outcomes, particularly in low-resource settings.

This solution leverages machine learning models trained on clinical data to classify maternal risk levels and provide real-time decision support for healthcare providers.
## Objectives
- Predict the likelihood of pre-eclampsia using clinical and demographic data.
- Develop interpretable models using techniques such as **SHAP** for explainability.
- Create a **decision support tool** that offers risk stratification and personalized healthcare suggestions.
- Evaluate model performance using multiple metrics to ensure clinical reliability and trust.
  
## Dataset

- Source: [Kaggle - Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data)
- Size: 1,014 samples
- Features:
  - Age
  - Systolic and Diastolic Blood Pressure
  - Blood Sugar
  - Body Temperature
  - Heart Rate
  - Risk Level (Target: Low, Medium, High)

## Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- AdaBoost
- Multi-Layer Perceptron (MLP)
- XGBoost

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

Best Performers:
- Random Forest: Accuracy = 0.92, F1 = 0.91, ROC AUC = 0.96
- XGBoost: Comparable high performance with strong class separation

## Key Tools & Techniques

- Python (Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn)
- SMOTE: For handling class imbalance
- StandardScaler: For normalization
- LabelEncoder: For encoding categorical target
- SHAP (SHapley Additive exPlanations): To interpret model predictions
- Matplotlib & Seaborn: For data visualization

## Project Highlights

- Balanced dataset using SMOTE to enhance fairness.
- Explainability through SHAP enables trust from clinicians.
- Data preprocessing, feature correlation, and outlier handling performed to ensure quality.
- Ensemble models proved most reliable for clinical prediction tasks.

## Future Work

- Deploy as a web or mobile-based decision support tool.
- Integrate with Electronic Health Records (EHR) for real-time clinical use.
- Expand dataset for external validation and generalization.
- Incorporate continuous patient monitoring for dynamic risk updates.

#Project Presentation (YouTube)
https://youtu.be/lrpo3K8k9no 

## Contributors
👩‍💻 Mary Nnipaa Meteku  
Group 22 — Artificial Intelligence in Healthcare

## License

This project is for academic and research purposes only. Contact the contributor for further use.

## Acknowledgments

- Kaggle for dataset access
- Researchers: Islam et al. (2022), Lundberg & Lee (2017), Chen & Guestrin (2016)
- Support from course instructor and peers
