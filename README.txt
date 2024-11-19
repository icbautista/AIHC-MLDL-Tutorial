Project Overview
-------------------------------
This project focuses on improving patient care using Machine Learning (ML) and Deep Learning (DL) models for Electronic Health Record (EHR) analysis, specifically targeting the detection of high-risk cases of suicidal ideation (SI) and suicide attempts (SA). The models used include Logistic Regression, Random Forest, and Bio_ClinicalBERT. This repository contains all the necessary code and resources for model training, evaluation, and a comprehensive analysis of performance metrics.

Presentation and Table of Contents
-------------------------------
* PDF not included.

Files in the Repository
-------------------------------
train_models.py: Contains the code for loading and preprocessing data, feature extraction using TF-IDF, and training ML models such as Logistic Regression and Random Forest.

evaluate_models.py: Includes the code for loading pre-trained models, evaluating model performance, and generating metrics such as accuracy, precision, recall, and confusion matrices.

results.txt: A text file summarizing the results of model evaluations, including key metrics and analysis outcomes.

Key Concepts and Methodologies
-------------------------------
Feature Engineering: TF-IDF Vectorization was used to convert text data into numerical features, enabling models to learn from the text.
ML Models:

Logistic Regression: Provides a simple and interpretable baseline for classification.
Random Forest: Captures complex, non-linear relationships in the data.

DL Model:
Bio_ClinicalBERT: Utilizes advanced natural language processing (NLP) techniques for understanding the context within clinical text.
Performance Metrics: Models were evaluated using metrics such as overall accuracy, precision, recall, and F1-score, with confusion matrices visualized as heatmaps to provide insights into model performance.

Visualizations
------------------
Model Performance Comparison: Bar charts and heatmaps are included to compare the performance of Logistic Regression, Random Forest, and Bio_ClinicalBERT across different metrics.
Heatmaps: Confusion matrices are visualized to show the distribution of true positives, false positives, true negatives, and false negatives for each model.

*PDF not included.

Future Enhancements
---------------------
* Explore additional feature engineering techniques to improve model performance.
* Experiment with more complex deep learning architectures for enhanced predictive capabilities.
* Conduct further analysis on model interpretability and clinical relevance.

Conclusion
----------
This project demonstrates the potential of ML and DL models in enhancing EHR analysis for early detection of high-risk cases. The comparative analysis highlights the strengths and weaknesses of each model, guiding their application in clinical practice to improve patient outcomes.

