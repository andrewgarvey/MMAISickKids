"""
Author: Andrew Garvey
Partner: Sargon Morad
Date: Aug 24, 2019
Client: Hospital for Sick Children

Title: Predict_DI_from_ED

Purpose:
-   Make Model for each modality, Random Forest and Logistic Regression
"""
# clear user created variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del name

# Basic Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_auc_score, roc_curve,  classification_report
#from sklearn.preprocessing import StandardScaler

import sklearn
sorted(sklearn.metrics.SCORERS.keys())
# set seed
Random_State = 42

# ----------------------------------------------------------------------------------------------------------------------
# Prep data splits
# Import data
ML_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/ML_Clean.csv')

# remove some columns that don't seem to be adding anything
ML_Clean = ML_Clean[ML_Clean.columns.drop(list(ML_Clean.filter(regex='Province|Arrived_|Method|Day_of_Arrival')))]

ML_Clean = ML_Clean.drop(['Age at Visit in days', 'Pulse Formatted', 'Resp Formatted', 'Temp Formatted',
                          'Gender_U', 'Encounter Number', 'Visits Since Aug 2018', 'Last Weight formatted'],axis=1)

"""
## Model based learning for additional data removing
# corr matrix
corr = ML_Clean.iloc[:,1:20].corr()
sns.heatmap(corr)
plt.show()
"""

#-------------------------------------------------------------------------------------------------------------
#General pre modeling

# Split data for modeling
Modalities = ['X-Ray', 'US', 'MRI', 'CT']

X = ML_Clean.drop(Modalities, axis=1)
y = ML_Clean[Modalities]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=Random_State)

# smote set up
sm=SMOTE(random_state=Random_State)
"""
# scaling (omitted to retain readability, actual implemented model should consider it)
scale = StandardScaler()

X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
"""
# ----------------------------------------------------------------------------------------------------------------------
"""
# Basic Random Forest
# Set initial directory

os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Model/Random_Forest')

# set params
grid_params_rf = [{'bootstrap': [True],
                   'criterion': ['entropy'],
                   'max_depth': [None,50],
                   'max_features': ['sqrt'],
                   'min_samples_leaf': [5],
                   'min_samples_split': [5],
                   'n_estimators': [1000]
                   }]
grid_cv_rf = 5
jobs_rf = 24



for index in range(0, len(Modalities)):

    # get the Modalities name for this loop
    Modality = Modalities[index]

    print("\n \n \n *********** Modality: " + Modality + " ***********")

    # set the y train with the target variable
    y_train_modality = y_train.iloc[:, y_train.columns == Modality].values.reshape(-1, )

    # original balance
    print('Pre-Smote: '+ str(Counter(y_train_modality)))

    #smote and new balance
    X_train_smote, y_train_modality_smote = sm.fit_resample(X_train, y_train_modality)
    print('Post-Smote: '+ str(Counter(y_train_modality_smote)))

    # Set the model conditions, run the model
    grid = GridSearchCV(estimator=RandomForestClassifier(random_state=Random_State), param_grid=grid_params_rf,
                        scoring='recall', cv=grid_cv_rf, n_jobs=jobs_rf, verbose=1)

    #grid.fit(X_train, np.ravel(y_train_modality))
    grid.fit(X_train_smote, np.ravel(y_train_modality_smote))

    # Evaluate training results
    print("*********** Training Results ***********")
    print("Best Roc Auc Score: " + str(grid.best_score_))
    print("Best Parameters: " + str(grid.best_params_))

    # Predict on Test Data
    pred_binary = grid.predict(X_test)
    pred = grid.predict_proba(X_test)
    pred_proba = pred[:, 1]
    y_test_modality = y_test.iloc[:, y_test.columns == Modality].values.reshape(-1, )

    # Evaluate Testing Results
    # binary
    print("*********** Binary Test Results ***********")
    print("Confusion Matrix: \n" + str(confusion_matrix(y_test_modality, pred_binary)))
    print("Classification Report:  \n" + str(classification_report(y_test_modality, pred_binary)))
    print("Accuracy: " + str(accuracy_score(y_test_modality, pred_binary)))

    # probabilistic
    print("*********** Probabilistic Test Results ***********")
    print("ROC AUC Score: \n" + str(roc_auc_score(y_test_modality, pred_proba)))

    # Auc Graph
    fpr, tpr, thresholds = roc_curve(y_test_modality, pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',  label='ROC curve (area = %0.2f)' % roc_auc)  # roc
    plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')  # baseline
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Random Forest " +str(Modality) + " ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
"""

# Logistic Regression
# Set initial directory
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Model/Logistic_Regression')

# Set all the variables

grid_params_lr = {'C': [0.1],
                  'solver': ["saga"],
                  'multi_class': ["multinomial"],
                  'max_iter':  [1000]}

grid_cv_lr = 3
jobs_lr = 24

LR_weights = pd.DataFrame(pd.Series(X.columns),columns=['Columns'])


for index in range(0, len(Modalities)):

    # get the Modalities name for this loop
    Modality = Modalities[index]

    print("\n \n \n *********** Modality: " + Modality + " ***********")

    # set the y train with the target variable
    y_train_modality = y_train.iloc[:, y_train.columns == Modality].values.reshape(-1, )

    # original balance
    print('Pre-Smote: '+ str(Counter(y_train_modality)))

    #smote and new balance
    X_train_smote, y_train_modality_smote = sm.fit_resample(X_train, y_train_modality)
    print('Post-Smote: '+ str(Counter(y_train_modality_smote)))
    
    # Set the model conditions, run the model
    grid = GridSearchCV(estimator=LogisticRegression(random_state=Random_State), param_grid=grid_params_lr,
                        scoring='roc_auc', cv=grid_cv_lr, n_jobs=jobs_lr, verbose=1)

    grid.fit(X_train_smote, np.ravel(y_train_modality_smote))

    # Evaluate training results
    print("*********** Training Results ***********")
    print("Best Roc Auc Score: " + str(grid.best_score_))
    print("Best Parameters: " + str(grid.best_params_))

    LR_weights[str(Modality)] = pd.Series((grid.best_estimator_.coef_)[0,:])

    # Predict on Test Data
    pred_binary = grid.predict(X_test)
    pred = grid.predict_proba(X_test)
    pred_proba = pred[:, 1]
    y_test_modality = y_test.iloc[:, y_test.columns == Modality].values.reshape(-1, )

    # Evaluate Testing Results
    # binary
    print("*********** Binary Test Results ***********")
    print("Confusion Matrix: \n" + str(confusion_matrix(y_test_modality, pred_binary)))
    print("Classification Report:  \n" + str(classification_report(y_test_modality, pred_binary)))
    print("Accuracy: " + str(accuracy_score(y_test_modality, pred_binary)))

    # probabilistic
    print("*********** Probabilistic Test Results ***********")
    print("ROC AUC Score: \n" + str(roc_auc_score(y_test_modality, pred_proba)))

    # Auc Graph
    fpr, tpr, thresholds = roc_curve(y_test_modality, pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',  label='ROC curve (area = %0.2f)' % roc_auc)  # roc
    plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')  # baseline
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Logistic Regression " + str(Modality) + " ROC Curve")
    plt.legend(loc="lower right")
    plt.show()



