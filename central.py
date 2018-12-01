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


#dataset description
d = train_data.describe()

#Filling missing values of Age
means_by_title = train_data[['Title', 'Age']].groupby('Title', sort= False).mean()
print(means_by_title)

features=['PassengerId','Name','Age']

temp = train_data[features]      # Copying ages data with Id
nan_rows = temp[temp.isnull().any(axis=1)]
print(nan_rows.loc[:, ['Age']])




print(pd.crosstab(train_data['Title'], train_data['Sex']))
print(pd.crosstab(train_data['Title'], train_data['Survived']))

# calculating means



