

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV


from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

data=pd.read_csv('dataset.csv', delimiter=',')
print(data.columns)


#Determine columns with missing values
for k in data.columns:
    n=data[k].count()
    if n<159541:
        print(k,": numberof missing values= ",159541-n) 
               
#Remove the 32 players with null PositionDesc
newdata=data.dropna()
col=data.columns


#Preprocessing criterion useful to decrease Knn complexity 
"""
newdata=data[(data.IntCaps+data.U21Caps)>0]
m=len(newdata)
newdata['Pos']=np.zeros(m)
newdata=newdata.reset_index()
"""
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


#Features after data visualization
selected_features=['Height', 'Weight', 'AerialAbility',
       'CommandOfArea', 'Communication', 'Eccentricity', 'Handling', 'Kicking',
       'OneOnOnes', 'Reflexes', 'RushingOut', 'TendencyToPunch', 'Throwing',
       'Corners', 'Crossing', 'Dribbling', 'Finishing', 'FirstTouch',
       'Freekicks', 'Heading', 'LongShots', 'Longthrows', 'Marking', 'Passing',
       'PenaltyTaking', 'Tackling', 'Technique', 'Aggression', 'Anticipation',
       'Bravery', 'Composure', 'Vision', 'Decisions',
        'Flair', 'Leadership', 'OffTheBall', 'Positioning',
       'Teamwork', 'Workrate', 'Acceleration', 'Agility', 'Balance', 'Jumping',
       'LeftFoot', 'NaturalFitness', 'Pace', 'RightFoot', 'Stamina',
       'Strength', 'Consistency', 'Dirtiness',  'Versatility',
       'Controversy']

X_data=newdata[selected_features]
y_data=newdata.Pos


#Dimensionality Reduction with PCA
data=X_data
M = np.mean(data,0) # compute the mean along columns axis=0 : M is a vector
C = data - M # subtract the mean (along columns)
W = np.dot(C.T, C) # compute covariance matrix
eigval,eigvec = np.linalg.eig(W) # compute eigenvalues and eigenvectors of covariance matrix
idx = eigval.argsort()[::-1] # Sort eigenvalues
eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues

#Determine the dimension of projection sub-space
    #calculate total variance
totalVariance=0
for k in range(len(idx)):
    totalVariance=totalVariance+idx[k]
    #choose the largest i eigenvalues such that variance/totalVariance>0.85
var=0
i=0    
while True:
    var=var+idx[i]
    i=i+1
    if var/totalVariance>0.85:
        break

#Project data to the new i dimensional subspace
X_data = np.dot(C,np.real(eigvec[:,:i]))

#Train_test split
X=X_data[:143092]
y=y_data[:143092]

test_data=X_data[143092:]
test_label=y_data[143092:]

# Decision tree
from sklearn.tree import DecisionTreeClassifier

#Hyperparameter Tuning
D=[10,11,12,14,16]
test_errors_DT=[] 
max_accuracy_DT=0
for j in range(len(D)):
    clf_dt = DecisionTreeClassifier(max_depth=D[j])
    DTscore= cross_val_score(clf_dt, X, y, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()
    if(DTscore>max_accuracy_DT):
        max_accuracy_DT=DTscore
        depth=D[j]# hyperparameter
    test_errors_DT.append(1-DTscore)

    #Plot test error vs hyperparameter tree depth
plt.plot(D, test_errors_DT)
plt.title("Test error vs. Decision Tree depth")
plt.ylabel("Test error")
plt.xlabel("tree depth")
plt.grid()
plt.show()


#K-Nearest neighbors 
from sklearn.neighbors import KNeighborsClassifier

#Hyperparameter Tuning  
K=[4,6,8,10,20,24,35]
test_errors_Knn=[] 
max_accuracy_Knn=0
for j in range(len(K)):
    clf_kn=KNeighborsClassifier(K[j])
    KNscore= cross_val_score(clf_kn, X, y, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()
    if(KNscore>max_accuracy_Knn):
        max_accuracy_Knn=KNscore # accuracy with best hyperparameter
        hp=K[j]# hyperparameter
    test_errors_Knn.append(1-KNscore)
    
    #Plot test error vs hyperparameter
plt.plot(K, test_errors_Knn)
plt.title("Test error vs. Nearest neighbors")
plt.ylabel("Test error")
plt.xlabel("Nearest neighbors")
plt.grid()
plt.show()


#Logistic regression
from sklearn.linear_model import LogisticRegression
clf_LR=LogisticRegression(solver='sag',multi_class='multinomial',C=0.5)
LR_score=cross_val_score(clf_LR, X, y, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()
clf_LR.fit(X,y)


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



#Evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
clf=[]
train_error=[]
Estimated_test_error=[]
test_error=[]

clf.append(LogisticRegression(solver='sag',multi_class='multinomial',C=0.5))
clf.append(  DecisionTreeClassifier(max_depth=12))
clf.append(KNeighborsClassifier(35))
Classifiers=['LR','DT','Knn']



for j in range(len(clf)):

    Estimated_test_error.append(cross_val_score(clf[j], X, y, scoring = 'accuracy', cv = 10, n_jobs = -1).mean())
    clf[j].fit(X, y)
    train_error.append(clf[j].score(X,y))
    test_error.append(clf[j].score(test_data,test_label))
    
    
#Error comparison
import matplotlib.pyplot as plt
#Estimated test error
estim = {'LR': Estimated_test_error[0], 'DT': Estimated_test_error[1],'Knn':Estimated_test_error[2]}
classifiers = list(estim)
e_t_error = list(estim.values())

#train error
train = {'LR': train_error[0], 'DT': train_error[1],'Knn': train_error[2]}

trainError = list(estim.values())

#test error
test = {'LR': test_error[0], 'DT': test_error[1], 'Knn': test_error[2]}

testEerror = list(estim.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axs[0].bar(classifiers, trainError)
axs[1].bar(classifiers,e_t_error)
axs[2].bar(classifiers,testEerror)
fig.suptitle('Error comparison')

initial_features=['Height', 'Weight', 'AerialAbility',
       'CommandOfArea', 'Communication', 'Handling', 'Kicking',
       'OneOnOnes', 'Reflexes', 'RushingOut', 'TendencyToPunch', 'Throwing',
       'Corners', 'Crossing', 'Dribbling', 'Finishing', 'FirstTouch',
       'Freekicks', 'Heading', 'LongShots', 'Longthrows', 'Marking', 'Passing',
       'PenaltyTaking', 'Tackling', 'Technique', 'Aggression', 'Anticipation',
       'Bravery', 'Composure', 'Concentration', 'Vision', 'Decisions',
       'Flair', 'Leadership', 'OffTheBall', 'Positioning',
       'Teamwork', 'Workrate', 'Acceleration', 'Agility', 'Balance', 'Jumping',
       'LeftFoot', 'NaturalFitness', 'Pace', 'RightFoot', 'Stamina',
       'Strength', 'Consistency', 'Dirtiness',
       'Versatility']


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


X_RFE=newdata[selected_features]
