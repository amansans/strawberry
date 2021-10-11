# strawberry

Data Source - https://www.kaggle.com/mlg-ulb/creditcardfraud

Data is skewed with a very small percentage of True positive cases of Credit Card Fraud. We want to build a Neural Net that predicts CreditCard Fraud.
Here I :
1) Visualize/preprocess data 
2) Explore Feature Selection/ Dimentionality Reduction
3) Build a basic Neural Net and try to tune HyperParams using SKopt
4) Build the final Neural Net using Tensorflow with Optuna as Hparam tuner

CreditCardFraudDetection_DataVizualisation_And_FeatureSelection.py - Credit Card Fraud Detection Dataset Visualisation and Feature Selection
CreditCardFraudDetection_TensorflowAndSkopt.py  - Creating a Neural Net and optimizing HyperParameters using SKOPT (Results not so great: Decent F1 score but overfitting)
CreditCardFraudDetection_TensorflowAndOptuna.py - Creating a Neural Net and optimizing HyperParameters using Optuna (Best Results: Eliminates overfitting and decent F1 score)
