📚 Library Performance Intelligence

An end-to-end machine learning–driven analytics system for analyzing, predicting, and segmenting public library performance using real-world operational data.
This project applies regression, classification, and clustering techniques and deploys them through an interactive analytics dashboard for practical decision support.

🔍 Project Overview

Public libraries generate large volumes of operational and financial data, yet most planning decisions rely on descriptive statistics and historical trends.
This project bridges that gap by applying predictive analytics and machine learning to transform raw public library data into actionable insights.

The system:

Predicts library circulation and operating income

Classifies libraries into funding levels

Segments libraries using unsupervised clustering

Presents insights through an interactive Streamlit dashboard

🎯 Objectives

Prepare and engineer high-quality features from multi-year public library data

Train and compare multiple regression and classification models

Identify the best-performing models using robust evaluation metrics

Discover latent patterns using K-Means clustering and PCA

Deploy models in a user-friendly analytics dashboard

🗂 Dataset

Source: Public Libraries Dataset (Connecticut)

Records: 5,278 libraries

Features: Demographic, operational, financial, and usage metrics

Preprocessing:

Missing value imputation

Feature scaling

Feature selection and engineering

🧠 Machine Learning Models
🔢 Regression

Used to predict Total Circulation and Operating Income

Linear Regression

Polynomial Regression

Decision Tree Regressor

Random Forest Regressor (Selected)

Evaluation Metrics: MAE, RMSE, R², Cross-Validation

🏷 Classification

Used to classify libraries into Funding Levels (Low / Medium / High)

Naive Bayes

K-Nearest Neighbors

Decision Tree

Support Vector Machine

Gradient Boosting Classifier (Selected)

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

📍 Clustering (Unsupervised Learning)

K-Means Clustering (k = 2 selected using Silhouette Score)

PCA for 2D visualization

Quality Metrics:

Silhouette Score

Davies–Bouldin Index

📊 Key Results

Random Forest Regression achieved R² > 0.96 for circulation prediction

Gradient Boosting Classifier achieved 93.7% accuracy in funding classification

Clustering revealed two distinct operational archetypes among libraries

Models demonstrated strong generalization through cross-validation

🖥 Interactive Dashboard

Built using Streamlit + Plotly, the dashboard provides:

📈 Real-time visual analytics

🔮 Single library prediction

📁 Bulk CSV predictions

📊 Model performance insights

📍 PCA-based clustering visualization

No machine learning knowledge is required to use the system.


├── Data_Cleaning.ipynb
├── EDA.ipynb
├── Regression1.ipynb
├── Classification1.ipynb
├── Clustering1.ipynb
├── app1.py
├── cleaned_data.csv
├── models/
│   ├── rf_circulation.pkl
│   ├── rf_income.pkl
│   ├── gb_classifier.pkl
│   ├── kmeans_model.pkl
│   └── scalers/
├── visuals/
│   └── saved_plots/
└── README.md


Run: 
streamlit run app1.py
