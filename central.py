import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV



#Load dataset
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
train_data.loc[train_data.Mr==1,'Age'] = train_data.loc[train_data.Mr==1,'Age'].fillna(MrAge)
train_data.loc[train_data.Mrs==1,'Age'] = train_data.loc[train_data.Mrs==1,'Age'].fillna(MrsAge)
train_data.loc[train_data.Miss==1,'Age'] = train_data.loc[train_data.Miss==1,'Age'].fillna(MissAge)
train_data.loc[train_data.Master==1,'Age'] = train_data.loc[train_data.Master==1,'Age'].fillna(MasterAge)
train_data.loc[train_data.Other==1,'Age'] = train_data.loc[train_data.Other==1,'Age'].fillna(OtherAge)


#Embarked Feature preprocessing
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

#Adding family category features
train_data['FamilyCateg']=train_data.SibSp+train_data.Parch
sns.factorplot('FamilyCateg',data=train_data,kind='count',hue='Survived')

train_data.FamilyCateg[train_data.FamilyCateg==0]=0
train_data.FamilyCateg[train_data.FamilyCateg>0]=1
train_data.FamilyCateg[train_data.FamilyCateg>3]=0
sns.factorplot('FamilyCateg',data=train_data,kind='count',hue='Survived')


#dataset description
d = train_data.describe()


print(pd.crosstab(train_data['Title'], train_data['Sex']))
print(pd.crosstab(train_data['Title'], train_data['Survived']))


##############################################################
###Features Selection RFE with the logistic regression estimator
features=['FamilyCateg','Age','Embarked_S','Embarked_Q','Embarked_C','Other','Mr','Mrs','Miss','Master','Fare','SibSp','Parch','Sex','Pclass']
X = train_data[features].values
Y = train_data['Survived'].values

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
estimator = LogisticRegression()
selector = RFECV(estimator, step=1, cv=StratifiedKFold(2) ,scoring='accuracy')
selector = selector.fit(X, Y)
print("Optimal number of features : %d" % selector.n_features_)
selected_features=[]
for i in range(len(features)):
    if selector.support_[i]==True:
        selected_features.append(features[i])
print(selected_features)
###########################################################""""

#Variable to display final scores of the following models
models=[]
final_scores=[]

#Logistic Regression

from sklearn.linear_model import LogisticRegression
X = train_data[selected_features].values
Y = train_data['Survived'].values
X_test = test_data[selected_features].values
Y_test= test_data['Survived'].values
clf = LogisticRegression().fit(X, Y)
Y_pred=clf.decision_function(X_test)

models.append('Logistic Regression')
final_scores.append(roc_auc)

#K-Nearest neighbors 
from sklearn.neighbors import KNeighborsClassifier
npX = np.array(train_data[selected_features]).copy()
npy = np.array(train_data['Survived']).copy()
    #Hyperparameter Tuning  
K=[1,2,3,4,5,6,7,8,9,10,11,12,13]
test_errors_Knn=[] 
max_accuracy_Knn=0
for j in range(len(K)):
    clf_kn=KNeighborsClassifier(K[j])
    KNscore= cross_val_score(clf_kn, npX, npy, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()
    if(KNscore>max_accuracy_Knn):
        max_accuracy_Knn=KNscore # accuracy with best hyperparameter
        hp=K[j]# hyperparameter
    test_errors_Knn.append(1-KNscore)
    
    #Plot test error vs hyperparameter
plt.plot(K, test_errors_Knn)
plt.title("Test error vs. Nearest neighbors")
plt.ylabel("Test error")
plt.xlabel("Nearest neighbors")
plt.show()
    #Add model and score
models.append('KNeighbors')
final_scores.append(max_accuracy_Knn)

# Decision tree
from sklearn.tree import DecisionTreeClassifier
npX = np.array(train_data[selected_features]).copy()
npy = np.array(train_data['Survived']).copy()
    #Hyperparameter Tuning
D=[2,4,6,8,10,12,14,16]
test_errors_DT=[] 
max_accuracy_DT=0
for j in range(len(D)):
    clf_dt = DecisionTreeClassifier(max_depth=D[j])
    DTscore= cross_val_score(clf_dt, npX, npy, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()
    if(DTscore>max_accuracy_DT):
        max_accuracy_DT=DTscore
        depth=D[j]# hyperparameter
    test_errors_DT.append(1-DTscore)

    #Plot test error vs hyperparameter tree depth
plt.plot(D, test_errors_DT)
plt.title("Test error vs. Decision Tree")
plt.ylabel("Test error")
plt.xlabel("tree depth")
plt.show()
    #Add model and score
models.append('Decision Tree')
final_scores.append(max_accuracy_DT)


#SVM (Soft Margin SVM)
from sklearn.svm import SVC
npX = np.array(train_data[selected_features]).copy()
npy = np.array(train_data['Survived']).copy()
    #Hyperparameter C Tuning
C=[0.5,1,2,4,6,8,10,20]
test_errors_svm=[] 
max_accuracy_svm=0
for j in range(len(C)):
    clf_svm = SVC(C = C[j])
    SVMscore= cross_val_score(clf_svm, npX, npy, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()
    if(SVMscore>max_accuracy_svm):
        max_accuracy_svm=SVMscore
        c=C[j]# hyperparameter
    test_errors_svm.append(1-SVMscore)
    
   #Plot test error vs hyperparameter c
plt.plot(C, test_errors_svm)
plt.title("Test error vs. SVM ")
plt.ylabel("Test error")
plt.xlabel("C parameter")
plt.show()
    #Add model and score
models.append('SVM (Soft Margin)')
final_scores.append(max_accuracy_svm)

# Display scores for the different models 
    #convert numpy array to pandas dataFrame
print( pd.DataFrame(final_scores, index = models, columns = ['accuracy']).sort_values(by = 'accuracy',ascending = False))
   
#Filling predicted labels into csv file
testId=test_data['PassengerId'].values
with open('results.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['PassengerId','Survived'])
    for i in range(mtest):    
        filewriter.writerow([testId[i],int(predictedLabels[i])])

##New
features=['FamilyCateg','Age','Embarked_S','Embarked_Q','Embarked_C','Other','Mr','Mrs','Miss','Master','Fare','SibSp','Parch','Sex','Pclass']
X = train_data[features]
Y = train_data['Survived']

def features_selection_RFE(estimator,X,Y):
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    selector = RFECV(estimator, step=1, cv=StratifiedKFold(2) ,scoring='accuracy')
    selector = selector.fit(X,Y)
    selected_features=[]
    for i in range(len(features)):
        if selector.support_[i]==True:
            selected_features.append(features[i])
    return(selected_features)
   
def cross_validation(k,X,Y,estimator):
    from sklearn.metrics import roc_curve, auc
    max_accuracy=0
    for i in range(k):
        X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=1/k, random_state=0)
        selected_features=features_selection_RFE(estimator,X_train,Y_train)
        new_X_train=X_train[selected_features]
        new_X_test=X_test[selected_features]
        estimator.fit(new_X_train,Y_train)
        Y_pred_i=estimator.decision_function(new_X_test)
        fpr,tpr,v = roc_curve(Y_test,Y_pred_i)
        if max_accuracy < auc(fpr,tpr):
            max_accuracy=auc(fpr,tpr)
    return (max_accuracy)
        
#max of acccuracy with the logistic regression
from sklearn.model_selection import train_test_split
X = train_data[features]
Y = train_data['Survived']
estimator=LogisticRegression()
print(cross_validation(30,X,Y,estimator))
       
