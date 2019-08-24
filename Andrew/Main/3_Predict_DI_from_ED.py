"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: Aug 24, 2019
Client: Hospital for Sick Children

Title: Predict_DI_from_ED

Purpose:
-   Turn cleaned data into usable ml data
"""
# clear user created variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del name

#Basic Imports
import numpy as np
import pandas as pd
import os

#Models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, \
                            roc_curve, recall_score, classification_report

# set seed
Random_State = 42

# ----------------------------------------------------------------------------------------------------------------------
# Prep data splits
# Import data
ML_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/ML_Clean.csv')

# Split data for modeling
Modalities = ['X-Ray', 'US', 'MRI', 'CT']

X = ML_Clean.drop(Modalities, axis=1)
y = ML_Clean[Modalities]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=Random_State)

# scaling
scale = StandardScaler()

X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# ----------------------------------------------------------------------------------------------------------------------
# Basic Random Forest
# Set initial directory
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Model/Random_Forest')

# set pipeline
pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state = 42))])

# set params
grid_params_rf = [{'clf__bootstrap': [True],
                   'clf__criterion': ['entropy'],
                   'clf__max_depth': [None],
                   'clf__max_features': [None],
                   'clf__min_samples_leaf': [2],
                   'clf__min_samples_split': [15],
                   'clf__n_estimators' : [100]
                   }]
grid_cv = 3
jobs = -1

# for index in range(0, len(Modalities)):

index = 0
# get the Modalities name for this loop
Modality = Modalities[index]

print("*********** Modality = " + Modality + " ***********")

# set the y train with the target variable
y_train_modality = y_train.iloc[:, y_train.columns == Modality].values.reshape(-1, )

# Set the model conditions, run the model
grid = GridSearchCV(estimator = pipe_rf, param_grid = grid_params_rf, scoring = 'roc_auc', cv = grid_cv, n_jobs = jobs,verbose = 1)

grid.fit(X_train,np.ravel(y_train_modality))

# Evaluate training results
print("*********** Training Results ***********")
print("Best Roc Auc Score: " + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))

# Predict on Test Data
pred_binary = grid.predict(X_test)
pred = grid.predict_proba(X_test)
pred_proba = pred[:,1]
y_test_modality = y_test.iloc[:, y_test.columns == Modality].values.reshape(-1, )

# Evaluate Testing Results
# binary
print("*********** Test Binary Results ***********")
print("Confusion Matrix: \n" + str(confusion_matrix(y_test_modality, pred_binary)))
print("Classification Report:  \n" + str(classification_report(y_test_modality,pred_binary)))
print("Accuracy: " + str(accuracy_score(y_test_modality,pred_binary)))

# probabilistic
print("*********** Test Probabilistic Results ***********")
roc_curve(y_test_modality,pred_proba)


# ----------------------------------------------------------------------------------------------------------------------
"""
# Logistic Regression
# Set initial directory
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Model/Logistic_Regression')

# Set all the hyper-variables
C = [0.01]
solver = ["saga"]
multi_class = ["multinomial"]
max_iter = [300]

parameters = {'C': C,
             'solver': solver,
             'multi_class': multi_class,
             'max_iter': max_iter}

# some loop that does random forest but for each of the aimed predictions
for index in range(0, len(Modalities)):


index = 0
# get the Modalities name for this loop
Modality = Modalities[index]

print("*********** Modality = " + Modality + " ***********")

# set the y train with the target variable
y_train_modality = y_train.iloc[:, y_train.columns == Modality].values.reshape(-1, )

print(">> finding best params")
LR_model = model_selection.GridSearchCV(linear_model.LogisticRegression(random_state=123),
                                   parameters, scoring="neg_log_loss",
                                   cv=2, n_jobs=-1, verbose=1)
LR_model.fit(X_train, y_train_modality)
best_params = LR_model.best_params_
print(">> best params: ", best_params)

# Predict probabilities
predicted_modality = LR_model.predict_proba(X)[:, 1]

# Compare to

print(">> saving validation info: ")
validation.to_csv(MODEL_NAME + "-" + tournament + ".csv")
print(">> done saving validation info")


# Check out results, in particular confusion matrix, I think what we aim for is a good ROC curve stats,
"""
