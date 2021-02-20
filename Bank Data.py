import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pylab
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn import metrics
# =============================================================================
# Que 2 :-Output variable -> y,  y -> Whether the client has subscribed a term deposit or not
# Binomial ("yes" or "no")
# =============================================================================

Bank = pd.read_csv("H:\DATA SCIENCE\Modules\Module 9 Logistic Regression\Datasets\Banks data.csv//bank_data.csv")
Bank.columns
Bank.info()
Bank.isnull().sum()
pd.set_option('display.max_columns',32)
pd.value_counts(Bank['y'].values)
Bank.shape

# Summary
Bank.describe()

# Converting numerical value in Binary of output
#Bank.y[Bank.y > 0] = 1

# Percentise
pd.value_counts(Bank['y'].values)
(39922/(5289+39922)) #0.88 %


######################## - Exploratary Data Analysis - ########################

# Measure of Central Tendancy / First moment business decision
Bank.mean()
Bank.median()
Bank.mode()

# Mesaure of Dispersion / Secound moment business decision
Bank.var()
Bank.std()

# Skewness / Kurtosis - Third and Forth moment business decision
Bank.skew()
Bank.kurt()

# Graphical Representaion
# Histogram
Bank.hist()
Bank.age.hist()

# Boxplot
plt.boxplot(Bank.y)
plt.boxplot(Bank.age)
# Heatmap plot
sns.heatmap(Bank.corr(),annot = True, annot_kws={"size": 5})

######################## - Spliting data in X and y - ########################

X = Bank.iloc[:,0:31]
y = Bank.iloc[:,31:32]

##################### - Spliting data in train and test - ####################

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 , random_state = False) # 30% Test data

##################### - Bulding logistic regression model - ####################

logit_model1 = smf.logit('y_train ~age+default+ balance+housing+loan+duration+campaign+pdays+previous+poutfailure+poutother+poutsuccess+poutunknown+con_cellular+con_telephone+con_unknown+divorced+married+single+joadmin+jobluecollar+joentrepreneur+johousemaid+jomanagement+joretired+joselfemployed+joservices+jostudent+jotechnician+jounemployed+jounknown', data = X_train).fit()
logit_model1.summary()
logit_model1.aic # 15660

##################### - Accuracy of model by test data - ####################

predict = logit_model1.predict(X_test)
predict

from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score

cnf_matrix = confusion_matrix(y_test, predict > 0.5 )
cnf_matrix

################## - Accuracy , Sensitivity , Specificity - #################

cnf_matrix
total1=sum(sum(cnf_matrix))
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy1) # 89 %

sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity1 ) ##0.97%

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

# Auc = 0.88












