# strawberry

Data Source - https://www.kaggle.com/mlg-ulb/creditcardfraud (Data is skewed with a very small percentage of True positive cases of Credit Card Fraud)

The goal was to create and train a Neural Network that detects Fraudulent Transactions. The following things were done to achieve this.
- Visualize/preprocess data (Mostly using Matplotlib/Seaborn and some Pandas data manipulation)
- Determining Outliers (Quartile Outliers/ Z score outliers)
- Explore Feature Selection/ Dimension Reduction (Using Anova/ Decision Trees/PCA) --> I dont use it for the final data that I train my model on
- Built a basic Neural Network and used SKOPT to tune Hyper Parameters (It didn't give me the results I wanted)
- Built the final Neural Network and used Optuna to tune the Hyper Parameters (Final F1 score around 88 and no overfitting)

Code for Visualization - CreditCardFraudDetection_DataVizualisation_And_FeatureSelection.py
Code with SKOPT- CreditCardFraudDetection_TensorflowAndSkopt.py
Code with Optuna- CreditCardFraudDetection_TensorflowAndOptuna.py

