# Predictive Maintenance Using Machine Learning: AI4I Dataset Analysis

*A full end-to-end data science workflow for predicting machine failures.*



##  Overview

This project explores the **AI4I 2020 Predictive Maintenance Dataset** to analyse machine operating conditions and build a classification model that predicts machine failures.  

The notebook demonstrates a complete data-science pipeline — from data cleaning and exploratory analysis to model training and evaluation.



Predictive maintenance is critical in modern industry, helping organisations reduce downtime, improve safety, and optimise maintenance costs.  

This project highlights the potential of machine-learning approaches in supporting these operational decisions.



---



##  Objectives



- Understand the structure and statistical properties of the AI4I dataset  

- Analyse operational conditions related to machine failures  

- Create visualisations to identify patterns and correlations  

- Build and evaluate a machine-learning model for failure prediction  

- Provide insights that support predictive maintenance strategies  



---



##  Key Steps Performed



###  **Data Cleaning & Preprocessing**

- Checked missing values  

- Encoded categorical features  

- Normalised/standardised inputs where needed  



###  **Exploratory Data Analysis**

- Distribution plots of operational parameters  

- Correlation heatmap  

- Failure type proportions  



### **Model Building**

- Train-test split  

- Random Forest classifier  

- Model evaluation using:  

&nbsp; - Accuracy  

&nbsp; - Confusion matrix  

&nbsp; - Classification report  



### **Results & Insights**

- Identification of operational conditions most associated with failure  

- Explanation of model performance  

- Visual summaries saved to the \*Results\* folder  



---



## Results Summary



The predictive model achieved strong performance in classifying machine failures, demonstrating the value of ML techniques in predictive maintenance.  

The analysis also revealed clear patterns in operational features such as \*\*torque\*\*, \*\*rotational speed\*\*, and \*\*tool wear\*\*, which significantly affect machine reliability.



---



##  Future Improvements



- Implement hyperparameter tuning (GridSearch/RandomSearch)  

- Add interpretability methods (SHAP values, feature importance)  

- Deploy the model via a Streamlit dashboard or API  

- Test the model using real-world industrial sensor data  


---



## Acknowledgements

Dataset Source:  

**UCI Machine Learning Repository — AI4I Predictive Maintenance Dataset**





