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


from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_auc_score, roc_curve,  classification_report

# ----------------------------------------------------------------------------------------------------------------------
# Global Variables

# Set output directory
os.chdir('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Model/Logistic_Regression')

# Import data
ML_Clean = pd.read_csv('/home/andrew/PycharmProjects/SickKidsMMAI/Generated_Outputs/Data/ML_Clean.csv')

# Set up total environment for model
X = ML_Clean.drop(Modalities, axis=1)
y = ML_Clean[Modalities]

# set seed
Random_State = 42

# smote set up
sm = SMOTE(random_state=Random_State)

# ----------------------------------------------------------------------------------------------------------------------
# Logistic Regression Variables
grid_params_lr = {'C': [0.1, 0.01, 0.001, 0.0001],
                  'solver': ['liblinear', "saga"],
                  'multi_class': ['ovr','auto'],
                  'max_iter':  [3000]}
grid_cv_lr = 10
jobs_lr = 20

# ----------------------------------------------------------------------------------------------------------------------
# Data Segmenting Variables

# Splits available
Modalities = ['Any', 'X-Ray', 'US', 'MRI', 'CT']
Ages = ['Any', 'Less1yr', '1-5yr', '6-10yr', 'Over10yr']
Genders = ['Any', 'F', 'M']

# grouping for age/gender specific data
Age_Grouping = pd.cut(ML_Clean['Age at Visit in days'],bins=(-10000, -1000, 365, 5*365, 10*365, 100*365), labels=Ages)
Gender_Grouping = pd.cut(ML_Clean['Gender_M'], bins=(-20, -1, 0.5, 2), labels=Genders)

# ----------------------------------------------------------------------------------------------------------------------
# Storing Looped Results
LR_weights = pd.DataFrame(pd.Series(X.columns), columns=['Columns'])

metrics = ['Modality', 'Age', 'Gender', 'DataSize', 'ROC_AUC', 'Accuracy', 'Confusion Matrix','Best Params']
LR_Metrics = pd.DataFrame(columns=metrics, index=range(0, len(Modalities)*len(Ages)*len(Genders)))

rowID = 0

# ----------------------------------------------------------------------------------------------------------------------
# Training a Model

for modality_index in range(0, len(Modalities)):
    for age_index in range(0, len(Ages)):
        for gender_index in range(0, len(Genders)):

            # get the Modalities/Age/Gender name for this loop
            modality = Modalities[modality_index]
            age = Ages[age_index]
            gender = Genders[gender_index]

            # use index to filter to specific X and y
            if (age == 'Any') & (gender == 'Any'):
                age_gender_index = Age_Grouping != 'bad age'  # all true
            elif age == 'Any':
                age_gender_index = (Gender_Grouping == gender)
            elif gender == 'Any':
                age_gender_index = (Age_Grouping == age)
            else:
                age_gender_index = (Age_Grouping == age) & (Gender_Grouping == gender)

            X_selected = X.loc[age_gender_index, :]
            y_selected = y.loc[age_gender_index, :]

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.25,
                                                                random_state=Random_State)

            print("\n \n *********** Mod:" + str(modality)+", Age:"+str(age)+", Gender:"+str(gender) + " ***********")

            # set the y train with the target variable
            y_train_modality = y_train.iloc[:, y_train.columns == modality].values.reshape(-1, )

            # original balance
            print('Pre-Smote: '+ str(Counter(y_train_modality)))

            # smote and new balance
            X_train_smote, y_train_modality_smote = sm.fit_resample(X_train, y_train_modality)
            print('Post-Smote: ' + str(Counter(y_train_modality_smote)))

            # Set the model conditions, run the model
            grid = GridSearchCV(estimator=LogisticRegression(random_state=Random_State), param_grid=grid_params_lr,
                                scoring='roc_auc', cv=grid_cv_lr, n_jobs=jobs_lr, verbose=1)

            grid.fit(X_train_smote, np.ravel(y_train_modality_smote))

            # Evaluate training results
            print("*********** Training Results ***********")
            print("Best Roc Auc Score: " + str(grid.best_score_))
            print("Best Parameters: " + str(grid.best_params_))

            # ----------------------------------------------------------------------------------------------------------
            # Predict on Test Data
            pred_binary = grid.predict(X_test)
            pred = grid.predict_proba(X_test)
            pred_proba = pred[:, 1]
            y_test_modality = y_test.iloc[:, y_test.columns == modality].values.reshape(-1, )

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
            plt.title("Logistic Regression- Mod:" + str(modality)+", Age:"+str(age)+", Gender:"+str(gender))
            plt.legend(loc="lower right")
            plt.savefig("Logistic Regression- Mod:" + str(modality)+", Age:"+str(age)+", Gender:" +str(gender) + ".pdf")

            # ----------------------------------------------------------------------------------------------------------
            # Store Weights
            LR_weights[str(modality) + " " + str(age) + " " + str(gender)] = pd.Series(grid.best_estimator_.coef_[0, :])

            # Store Metrics
            metric_answers = [modality,
                              age,
                              gender,
                              len(y_selected),
                              roc_auc_score(y_test_modality, pred_proba),
                              accuracy_score(y_test_modality, pred_binary),
                              str(confusion_matrix(y_test_modality, pred_binary)),
                              str(grid.best_params_)
                              ]

            for metric_index in range(0, len(metric_answers)):
                LR_Metrics.iloc[rowID, LR_Metrics.columns == metrics[metric_index]] = metric_answers[metric_index]

            # Increment Row for next loop
            rowID = rowID+1

# Store end results as csv
LR_Metrics.to_csv('LR_Metrics.csv')
LR_weights.to_csv('LR_weights.csv')


