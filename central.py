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

#dataset description
d = train_data.describe()


print(pd.crosstab(train_data['Title'], train_data['Sex']))
print(pd.crosstab(train_data['Title'], train_data['Survived']))

# calculating means



