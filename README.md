# Predictive Maintenance: Machine Failure Classification

A featured machine learning project that predicts machine failure risk using the AI4I 2020 Predictive Maintenance dataset.

## Project Overview

Unexpected machine failure can lead to costly downtime, disrupted operations, and higher maintenance expenses. This project builds a predictive maintenance workflow that uses machine operating data to identify failure risk before breakdown occurs.

Using the **AI4I 2020 Predictive Maintenance Dataset**, I developed an end-to-end machine learning pipeline covering data preparation, feature engineering, model comparison, threshold tuning, evaluation, and interactive deployment.

Rather than relying on a single model, this project compares multiple classification approaches and selects the strongest model based on practical predictive-maintenance metrics such as **F1-score**, **recall**, **ROC-AUC**, and **average precision**.

## Business Problem

In industrial settings, reactive maintenance is expensive and inefficient. The aim of this project is to support **proactive maintenance planning** by identifying machines with a higher probability of failure using their operating conditions and wear-related signals.

This kind of system can help maintenance teams:
- reduce unplanned downtime
- prioritise inspections
- improve asset reliability
- make maintenance decisions earlier

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains machine operating and product-related features such as:

- air temperature
- process temperature
- rotational speed
- torque
- tool wear
- product type

Target variable:
- **Machine failure** (binary classification)

Dataset source: **UCI Machine Learning Repository**

## Project Objectives

- understand the structure and failure patterns in the dataset
- explore which machine conditions are associated with failure
- engineer useful features for predictive modelling
- compare multiple machine learning models
- evaluate performance using failure-focused metrics
- support predictive maintenance decisions with interpretable outputs

## Workflow

## 1. Data Cleaning and Preparation
- checked data types and missing values
- removed identifiers and leakage-related columns where needed
- prepared numeric and categorical preprocessing pipelines
- applied scaling, imputation, and encoding through a reproducible pipeline

## 2. Feature Engineering
To strengthen the modelling workflow, I created additional features such as:
- temperature difference
- power proxy
- torque-wear interaction
- wear-to-speed ratio

These features were designed to capture machine stress and operating relationships more effectively than raw inputs alone.

## 3. Exploratory Data Analysis
The notebook includes:
- target distribution analysis
- operational parameter distributions
- correlation heatmap
- failure-rate pattern analysis across feature ranges
- visual comparison of machine conditions linked to higher risk

## 4. Model Comparison
I compared multiple classification models rather than assuming one model would automatically perform best:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

This made the final model choice more credible and better aligned with the project objective.

## 5. Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix
- ROC and Precision-Recall curves

Because machine failure is the key business risk, I placed particular attention on **recall**, **F1-score**, and **probability-based evaluation** rather than accuracy alone.

## 6. Threshold Tuning
The project also explores decision-threshold tuning to show how the balance between **precision** and **recall** changes.

This is important in predictive maintenance because the cost of missing a real failure may be much higher than the cost of investigating a false alarm.

## Final Model Performance

The best-performing model in this project was **Gradient Boosting**.

### Holdout Results
- **F1-score:** 0.885
- **Recall:** 0.794
- **Precision:** 1.000
- **ROC-AUC:** 0.971
- **Average Precision:** 0.900
- **Best threshold by F1:** 0.45

These results show that the selected model achieved strong ranking quality and strong failure-class performance while maintaining very high precision.

## Key Findings

- Gradient Boosting outperformed the other tested models on the holdout set.
- Failure prediction improved when the workflow moved beyond a single-model approach.
- Operating conditions related to **torque**, **rotational speed**, **tool wear**, and engineered stress-related features showed strong relationships with failure risk.
- Threshold tuning showed that the default 0.50 probability cut-off was not necessarily the best operating point.
- The project works best as a **failure-risk ranking and prioritisation tool**, not just a binary classifier.

## Streamlit Application

This project also includes a **Streamlit decision-support dashboard** for interactive use.

The app allows users to:
- explore the dataset
- compare models
- inspect evaluation metrics
- view feature importance
- test custom machine scenarios
- estimate failure probability and risk level

This makes the project more practical and portfolio-ready by translating model outputs into a usable interface.

## Repository Structure

```text
Predictive-Maintenance-Project/
│
├── README.md
├── LICENSE
├── requirements.txt
├── app.py
├── predictive_maintenance.ipynb
├── data.csv
└── results/
```

## Acknowledgements

Dataset source:
UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset

## Author

Owen Nda Diche