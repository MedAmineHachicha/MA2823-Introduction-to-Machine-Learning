import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import sklearn

train_data=pd.read_csv('train.csv', delimiter=',')
test_data=pd.read_csv('test.csv', delimiter=',')

#changing Sex to binary
train_data.Sex[train_data.Sex == 'male'] = 0
train_data.Sex[train_data.Sex == 'female'] = 1

#adding column for Title
train_data['Title'] = train_data.Name.str.extract('([A-Za-z]+)\.', expand = True)
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

#Making title categories
    #Mr Category
train_data['Mr']=train_data.Title[train_data.Title == 'Mr']
train_data.Mr[train_data.Mr == 'Mr'] = 1
train_data.Mr[train_data.Mr != 1] = 0
    #Mrs Category
train_data['Mrs']=train_data.Title[train_data.Title == 'Mrs']
train_data.Mrs[train_data.Mrs == 'Mrs'] = 1
train_data.Mrs[train_data.Mrs != 1] = 0
    #Master Category
train_data['Master']=train_data.Title[train_data.Title == 'Master']
train_data.Master[train_data.Master == 'Master'] = 1
train_data.Master[train_data.Master != 1] = 0
    #Miss Category
train_data['Miss']=train_data.Title[train_data.Title == 'Miss']
train_data.Miss[train_data.Miss == 'Miss'] = 1
train_data.Miss[train_data.Miss != 1] = 0
    #Others
train_data['Other']=1-(train_data.Mr+train_data.Mrs+train_data.Miss+train_data.Master)


#Age feature preprocessing
    #Calculating age mean by title category
MrAge=train_data.Age[train_data.Mr==1].mean()
MasterAge=train_data.Age[train_data.Master==1].mean()
MrsAge=train_data.Age[train_data.Mrs==1].mean()
MissAge=train_data.Age[train_data.Miss==1].mean()
OtherAge=train_data.Age[train_data.Other==1].mean()

    #Filling missing values of Age by title category
train_data.Age[train_data.Age.isnull().any(axis=0) and train_data.Mr==1]=MrAge
train_data.Age[train_data.Age.isnull().any(axis=0) and train_data.Mrs==1]=MrsAge
train_data.Age[train_data.Age.isnull().any(axis=0) and train_data.Miss==1]=MissAge
train_data.Age[train_data.Age.isnull().any(axis=0) and train_data.Master==1]=MasterAge
train_data.Age[train_data.Age.isnull().any(axis=0) and train_data.Other==1]=OtherAge

#Embarked Feature 
    #filling nan with S( less impact on survival rate)
train_data['Embarked']=train_data['Embarked'].fillna('S')

    #Making Embarked categories
    #Embarked_S
train_data['Embarked_S']=train_data.Embarked[train_data.Embarked== 'S']
train_data.Embarked_S[train_data.Embarked_S == 'S'] = 1
train_data.Embarked_S[train_data.Embarked_S != 1] = 0
     #Embarked_Q
train_data['Embarked_Q']=train_data.Embarked[train_data.Embarked == 'Q']
train_data.Embarked_Q[train_data.Embarked_Q == 'Q'] = 1
train_data.Embarked_Q[train_data.Embarked_Q != 1] = 0
     #Embarked_S
train_data['Embarked_C']=train_data.Embarked[train_data.Embarked == 'C']
train_data.Embarked_C[train_data.Embarked_C == 'C'] = 1
train_data.Embarked_C[train_data.Embarked_C != 1] = 0

#dataset description
d = train_data.describe()


print(pd.crosstab(train_data['Title'], train_data['Sex']))
print(pd.crosstab(train_data['Title'], train_data['Survived']))

# calculating means


#Logistic Regression
features=['Age','Embarked_S','Embarked_Q','Embarked_C','Other','Mr','Mrs','Miss','Master','Fare','SibSp','Parch','Sex','Pclass']
X = train_data[features].values
Y = train_data['Survived'].values
X = test_data[features].values
Y = test_data['Survived'].values
clf = LogisticRegression().fit(X, Y)
Y_pred=clf.decision_function(X_test)
fpr,tpr,_ = roc_curve(Y_test,Y_pred)
roc_auc = auc(fpr,tpr)

#Features Selection RFE with cross-validation 
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selected_features=[]
for i in range(len(features)):
    if Selector.support_[i]==True:
        selected_features.append(features[i])
print(selected_features)


