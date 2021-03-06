# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:16:51 2020

@author: Deepak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:38:40 2020
@author: DESHMUKH
LOGISTIC REGRESSION ASSIGNMENT
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pylab
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn import metrics
# ==================================================================================================
# Que 1 :- The predictor variables of interest are the amount of money spent on the campaign,
##    the amount of time spent campaigning negatively and whether or not the candidate is an incumbent
# ======================================================================================================

election = pd.read_csv("H:\DATA SCIENCE\Modules\Module 9 Logistic Regression\Datasets\election_data.csv//election_data.csv")
election.columns
election.info()
election.isnull().sum()
pd.set_option('display.max_columns',20)
pd.value_counts(election['Result'].values)
election.shape

# Summary
election.describe()

# Converting numerical value in Binary of output
##election.nelection[election.nelection > 0] = 1

# Percentise
pd.value_counts(election['Result'].values)
(4/(4+6)) #40 %


######################## - Exploratary Data Analysis - ########################

# Measure of Central Tendancy / First moment business decision
election.mean()
election.median()
election.mode()

# Mesaure of Dispersion / Secound moment business decision
election.var()
election.std()

# Skewness / Kurtosis - Third and Forth moment business decision
election.skew()
election.kurt()

# Graphical Representaion
# Histogram
election.hist()

# Boxplot
plt.boxplot(election.Result)

# Heatmap plot
sns.heatmap(election.corr(),annot = True, annot_kws={"size": 5})

######################## - Spliting data in X and y - ########################

X = election.iloc[:,2:5]
X
y = election.iloc["result"]
y

##################### - Spliting data in train and test - ####################

 #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = False) # 30% Test data

##################### - Bulding logistic regression model - ####################
model = smf.logit('y ~ amountspend +popularityrank', data =X ).fit()
model.summary()
model.aic # aci=6
logit_model1 = smf.logit('y ~  amountspend', data = X).fit()
logit_model1.summary()
logit_model1.aic # 8.33

##################### - Accuracy of model by test data - ####################

predict = model.predict(X)
predict

from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score

print(confusion_matrix(y, predict > 0.5 ))


################## - Accuracy , Sensitivity , Specificity - #################

cnf_matrix
total1=sum(sum(cnf_matrix))
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy1) # 79 %

sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : ', specificity1)

######################## -  ROC Curve - #######################

fpr, tpr, threshold = metrics.roc_curve(y_test, predict)
roc_auc = metrics.auc(fpr, tpr)

#import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Auc = 0.96












