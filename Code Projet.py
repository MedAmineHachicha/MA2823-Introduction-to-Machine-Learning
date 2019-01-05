import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import sklearn
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

data=pd.read_csv('dataset.csv', delimiter=',')
print(data.columns)

newdata=data[(data.IntCaps+data.U21Caps)>0]
m=len(newdata)
newdata['Pos']=np.zeros(m)
newdata=newdata.reset_index()

#Creating target labels with 7 different positions on football pitch
for i in range(len(newdata)):
    if newdata.Goalkeeper.iloc[i]==20:
        newdata.at[i,'Pos']=1
    if newdata.Sweeper.iloc[i]==20 or newdata.DefenderCentral.iloc[i]==20:
        newdata.at[i,'Pos']=2
    if newdata.DefenderLeft.iloc[i]==20 or newdata.DefenderRight.iloc[i]==20 or newdata.WingBackLeft.iloc[i]==20 or newdata.WingBackRight.iloc[i]==20:
        newdata.at[i,'Pos']=3
    if newdata.DefensiveMidfielder.iloc[i]==20 or newdata.MidfielderCentral.iloc[i]==20:
        newdata.at[i,'Pos']=4
    if newdata.Striker.iloc[i]==20:
        newdata.at[i,'Pos']=5
    if newdata.MidfielderRight.iloc[i]==20 or newdata.MidfielderLeft.iloc[i]==20 or newdata.AttackingMidLeft.iloc[i]==20 or newdata.AttackingMidRight.iloc[i]==20:
        newdata.at[i,'Pos']=6
    if newdata.AttackingMidCentral.iloc[i]==20:
        newdata.at[i,'Pos']=7

#Shuffle Data        
newdata = shuffle(newdata)

Columns=['Height', 'Weight', 'AerialAbility',
       'CommandOfArea', 'Communication', 'Eccentricity', 'Handling', 'Kicking',
       'OneOnOnes', 'Reflexes', 'RushingOut', 'TendencyToPunch', 'Throwing',
       'Corners', 'Crossing', 'Dribbling', 'Finishing', 'FirstTouch',
       'Freekicks', 'Heading', 'LongShots', 'Longthrows', 'Marking', 'Passing',
       'PenaltyTaking', 'Tackling', 'Technique', 'Aggression', 'Anticipation',
       'Bravery', 'Composure', 'Concentration', 'Vision', 'Decisions',
       'Determination', 'Flair', 'Leadership', 'OffTheBall', 'Positioning',
       'Teamwork', 'Workrate', 'Acceleration', 'Agility', 'Balance', 'Jumping',
       'LeftFoot', 'NaturalFitness', 'Pace', 'RightFoot', 'Stamina',
       'Strength', 'Consistency', 'Dirtiness', 'ImportantMatches',
       'InjuryProness', 'Versatility', 'Adaptability', 'Ambition', 'Loyalty',
       'Pressure', 'Professional', 'Sportsmanship', 'Temperament',
       'Controversy','Pos']

finaldata=newdata[Columns]

initial_features=['Height', 'Weight', 'AerialAbility',
       'CommandOfArea', 'Communication', 'Eccentricity', 'Handling', 'Kicking',
       'OneOnOnes', 'Reflexes', 'RushingOut', 'TendencyToPunch', 'Throwing',
       'Corners', 'Crossing', 'Dribbling', 'Finishing', 'FirstTouch',
       'Freekicks', 'Heading', 'LongShots', 'Longthrows', 'Marking', 'Passing',
       'PenaltyTaking', 'Tackling', 'Technique', 'Aggression', 'Anticipation',
       'Bravery', 'Composure', 'Concentration', 'Vision', 'Decisions',
       'Determination', 'Flair', 'Leadership', 'OffTheBall', 'Positioning',
       'Teamwork', 'Workrate', 'Acceleration', 'Agility', 'Balance', 'Jumping',
       'LeftFoot', 'NaturalFitness', 'Pace', 'RightFoot', 'Stamina',
       'Strength', 'Consistency', 'Dirtiness', 'ImportantMatches',
       'InjuryProness', 'Versatility', 'Adaptability', 'Ambition', 'Loyalty',
       'Pressure', 'Professional', 'Sportsmanship', 'Temperament',
       'Controversy']
X=newdata[initial_features]
y=newdata.Pos


#Feature Selection with RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
estimator = LogisticRegression()
selector = RFECV(estimator, step=1, cv=StratifiedKFold(2) ,scoring='accuracy')
selector = selector.fit(X, y)
print("Optimal number of features : %d" % selector.n_features_)
selected_features=[]
for i in range(len(initial_features)):
    if selector.support_[i]==True:
        selected_features.append(initial_features[i])
print(selected_features)

train_data=X[:11000]
test_data=X[11000:]
train_labels=y[:11000]
test_labels=y[11000:]


#Hyperparameter tuning for kNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

accuracy=[]
for k in range(1,50):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_data,train_labels)
    predicted_labels=neigh.predict(test_data)
    acc=accuracy_score(test_labels, predicted_labels)
    accuracy.append(acc)
    print(acc)
plt.plot(accuracy)

# Data Visualisation (replace Tackling with any other feature)
categ=[2,3,4,5,6,7]
Pos=['CB','WB','CMD','ST','WMD','AMD']
for cat in categ:
    subset = newdata[newdata['Pos'] == cat]
    
    sns.distplot(subset['Tackling'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2},
                 label = Pos[cat-2])
    
# Plot formatting
plt.legend(prop={'size': 10}, title = 'Positions')
plt.title('Density Plot for Tackling')
plt.xlabel('Player Attribute')
plt.ylabel('Density')