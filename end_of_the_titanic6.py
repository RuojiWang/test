#coding=utf-8
import os
import sys
import random
import pickle
import datetime
import numpy as np
import pandas as pd
sys.path.append("D:\\Workspace\\Titanic")
from Utilities1 import noise_augment_pytorch_classifier

from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, StratifiedKFold

import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

import skorch
from skorch import NeuralNetClassifier

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


data_train = pd.read_csv("C:/Users/1/Desktop/train.csv")
data_test = pd.read_csv("C:/Users/1/Desktop/test.csv")
combine = [data_train, data_test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    dataset['Title'] = dataset['Title'].map(title_map)
    dataset['Title'] = dataset['Title'].fillna(0)   

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['FamilySizePlus'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizePlus'] = 1
    dataset.loc[dataset['FamilySize'] == 2, 'FamilySizePlus'] = 2
    dataset.loc[dataset['FamilySize'] == 3, 'FamilySizePlus'] = 2
    dataset.loc[dataset['FamilySize'] == 4, 'FamilySizePlus'] = 2
    dataset.loc[dataset['FamilySize'] == 5, 'FamilySizePlus'] = 1
    dataset.loc[dataset['FamilySize'] == 6, 'FamilySizePlus'] = 1
    dataset.loc[dataset['FamilySize'] == 7, 'FamilySizePlus'] = 1

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)
    
for dataset in combine: 
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0 
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1 
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2 
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3 
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
#杩欓噷鐨刴ode鏄眰瑙andas.core.series.Series浼楁暟鐨勭涓�涓�硷紙鍙兘鏈夊涓紬鏁帮級
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#灏哾ata_test涓殑fare鍏冪礌鎵�缂哄け鐨勯儴鍒嗙敱宸茬粡鍖呭惈鐨勬暟鎹殑涓綅鏁板喅瀹氬搱
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in combine:
    dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = 0
    dataset.loc[(dataset.Cabin.notnull()), 'Cabin'] = 1

#灏肩帥缁欎綘璇寸殑杩欎釜鏄础鐚埞绁紝鍘熸潵鐨勮嫳鏂囬噷闈㈡牴鏈氨娌℃湁杩欑璇存硶鍢�
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
#print(df)
df_ticket = df.index.values          #鍏变韩鑸圭エ鐨勭エ鍙�
tickets = data_train.Ticket.values   #鎵�鏈夌殑鑸圭エ
#print(tickets)
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #閬嶅巻鎵�鏈夎埞绁紝鍦ㄥ叡浜埞绁ㄩ噷闈㈢殑涓�1锛屽惁鍒欎负0
    result.append(ticket)
    
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values          #鍏变韩鑸圭エ鐨勭エ鍙�
tickets = data_train.Ticket.values   #鎵�鏈夌殑鑸圭エ

result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #閬嶅巻鎵�鏈夎埞绁紝鍦ㄥ叡浜埞绁ㄩ噷闈㈢殑涓�1锛屽惁鍒欎负0
    result.append(ticket)

results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_train = pd.concat([data_train, results], axis=1)

df = data_test['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values          
tickets = data_test.Ticket.values   
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   
    result.append(ticket)
results = pd.DataFrame(result)
results.columns = ['Ticket_Count']
data_test = pd.concat([data_test, results], axis=1) 

data_train_1 = data_train.copy()
data_test_1  = data_test.copy()
data_test_1 = data_test_1.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis=1)

X_train = data_train_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]
Y_train = data_train_1['Survived']

X_test = data_test_1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Cabin', 'Title', 'FamilySizePlus', 'Ticket_Count']]

X_all = pd.concat([X_train, X_test], axis=0)
#鎴戣寰楄缁冮泦鍜屾祴璇曢泦闇�瑕佸湪涓�璧疯繘琛岀壒寰佺缉鏀撅紝鎵�浠ユ敞閲婃帀浜嗗師鏉ョ殑X_train鐨勭壒寰佺缉鏀惧挴
X_all_scaled = pd.DataFrame(preprocessing.scale(X_all), columns = X_train.columns)
#X_train_scaled = pd.DataFrame(preprocessing.scale(X_train), columns = X_train.columns)
X_train_scaled = X_all_scaled[:len(X_train)]
X_test_scaled = X_all_scaled[len(X_train):]

class MyModule1(nn.Module):
    def __init__(self):
        super(MyModule1, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc1.bias.data, 0)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self   
    
class MyModule2(nn.Module):
    def __init__(self):
        super(MyModule2, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self 
    
class MyModule3(nn.Module):
    def __init__(self):
        super(MyModule3, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self 

class MyModule4(nn.Module):
    def __init__(self):
        super(MyModule4, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.softmax(self.fc3(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
        
        return self 

class MyModule5(nn.Module):
    def __init__(self):
        super(MyModule5, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self   
    
class MyModule6(nn.Module):
    def __init__(self):
        super(MyModule6, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self
    
class MyModule7(nn.Module):
    def __init__(self):
        super(MyModule7, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self

class MyModule8(nn.Module):
    def __init__(self):
        super(MyModule8, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))
        X = F.softmax(self.fc4(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
        
        return self

class MyModule9(nn.Module):
    def __init__(self):
        super(MyModule9, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self   
    
            
class MyModule10(nn.Module):
    def __init__(self):
        super(MyModule10, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self

class MyModule11(nn.Module):
    def __init__(self):
        super(MyModule11, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self

class MyModule12(nn.Module):
    def __init__(self):
        super(MyModule12, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = F.relu(self.fc3(X))   
        X = self.dropout2(X)
        X = F.relu(self.fc4(X))  
        X = F.softmax(self.fc5(X), dim=-1)
        return X

    def init_weight(self, mode):
        if (mode==1):
            pass#灏辨槸浣跨敤榛樿璁剧疆鐨勬剰鎬濆挴
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
    
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
        
        return self
       
class MyModule13(nn.Module):
    def __init__(self):
        super(MyModule13, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self
    
class MyModule14(nn.Module):
    def __init__(self):
        super(MyModule14, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self

class MyModule15(nn.Module):
    def __init__(self):
        super(MyModule15, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 60)
        self.fc6 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self
    
class MyModule16(nn.Module):
    def __init__(self):
        super(MyModule16, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 70)
        self.fc6 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.softmax(self.fc6(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            
        return self
    
class MyModule17(nn.Module):
    def __init__(self):
        super(MyModule17, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, 40)
        self.fc7 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self
    
class MyModule18(nn.Module):
    def __init__(self):
        super(MyModule18, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self

class MyModule19(nn.Module):
    def __init__(self):
        super(MyModule19, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 60)
        self.fc6 = nn.Linear(60, 60)
        self.fc7 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self
        
class MyModule20(nn.Module):
    def __init__(self):
        super(MyModule20, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 70)
        self.fc6 = nn.Linear(70, 70)
        self.fc7 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.softmax(self.fc7(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            
        return self
    
class MyModule21(nn.Module):
    def __init__(self):
        super(MyModule21, self).__init__()

        self.fc1 = nn.Linear(9, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, 40)
        self.fc6 = nn.Linear(40, 40)
        self.fc7 = nn.Linear(40, 40)
        self.fc8 = nn.Linear(40, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
class MyModule22(nn.Module):
    def __init__(self):
        super(MyModule22, self).__init__()

        self.fc1 = nn.Linear(9, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
class MyModule23(nn.Module):
    def __init__(self):
        super(MyModule23, self).__init__()

        self.fc1 = nn.Linear(9, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, 60)
        self.fc5 = nn.Linear(60, 60)
        self.fc6 = nn.Linear(60, 60)
        self.fc7 = nn.Linear(60, 60)
        self.fc8 = nn.Linear(60, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
class MyModule24(nn.Module):
    def __init__(self):
        super(MyModule24, self).__init__()

        self.fc1 = nn.Linear(9, 70)
        self.fc2 = nn.Linear(70, 70)
        self.fc3 = nn.Linear(70, 70)
        self.fc4 = nn.Linear(70, 70)
        self.fc5 = nn.Linear(70, 70)
        self.fc6 = nn.Linear(70, 70)
        self.fc7 = nn.Linear(70, 70)
        self.fc8 = nn.Linear(70, 2)  
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout1(X)
        X = F.relu(self.fc4(X))  
        X = self.dropout1(X)
        X = F.relu(self.fc5(X))
        X = F.relu(self.fc6(X))
        X = F.relu(self.fc7(X))
        X = F.softmax(self.fc8(X), dim=-1)
        return X

    def init_weight(self, mode):
        
        if (mode==1):
            pass#灏辨槸浠�涔堥兘涓嶅仛鐨勬剰鎬�
        
        elif (mode==2):
            torch.nn.init.normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        elif (mode==3):
            torch.nn.init.xavier_normal_(self.fc1.weight.data)
            torch.nn.init.constant_(self.fc1.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc2.weight.data)
            torch.nn.init.constant_(self.fc2.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc3.weight.data)
            torch.nn.init.constant_(self.fc3.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc4.weight.data)
            torch.nn.init.constant_(self.fc4.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc5.weight.data)
            torch.nn.init.constant_(self.fc5.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc6.weight.data)
            torch.nn.init.constant_(self.fc6.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc7.weight.data)
            torch.nn.init.constant_(self.fc7.bias.data, 0)
            torch.nn.init.xavier_normal_(self.fc8.weight.data)
            torch.nn.init.constant_(self.fc8.bias.data, 0)
        
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight.data)
            torch.nn.init.xavier_uniform_(self.fc2.weight.data)
            torch.nn.init.xavier_uniform_(self.fc3.weight.data)
            torch.nn.init.xavier_uniform_(self.fc4.weight.data)
            torch.nn.init.xavier_uniform_(self.fc5.weight.data)
            torch.nn.init.xavier_uniform_(self.fc6.weight.data)
            torch.nn.init.xavier_uniform_(self.fc7.weight.data)
            torch.nn.init.xavier_uniform_(self.fc8.weight.data)
            
        return self
    
module1 = MyModule1()
module2 = MyModule2()    
module3 = MyModule3()
module4 = MyModule4()
module5 = MyModule5()
module6 = MyModule6()
module7 = MyModule7()
module8 = MyModule8()
module9 = MyModule9()
module10 = MyModule10()    
module11 = MyModule11()
module12 = MyModule12()
module13 = MyModule13()
module14 = MyModule14()
module15 = MyModule15()
module16 = MyModule16()
module17 = MyModule17()
module18 = MyModule18()    
module19 = MyModule19()
module20 = MyModule20()
module21 = MyModule21()
module22 = MyModule22()
module23 = MyModule23()
module24 = MyModule24()

net = NeuralNetClassifier(
    module = module3,
    lr=0.1,
    #device="cuda",
    device="cpu",
    max_epochs=400,
    #criterion=torch.nn.NLLLoss,
    optimizer=torch.optim.Adam,
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=[skorch.callbacks.EarlyStopping(patience=10)]
)

def cal_nnclf_acc(clf, X_train, Y_train):
    
    Y_train_pred = clf.predict(X_train.values.astype(np.float32))
    count = (Y_train_pred == Y_train).sum()
    acc = count/len(Y_train)
    
    return acc

def print_nnclf_acc(acc):
    
    print("the accuracy rate of the model on the whole train dataset is:", acc)
  
def print_best_params_acc(trials):
    
    trials_list =[]
    #浠巘rials涓鍙栨渶澶х殑鍑嗙‘鐜囦俊鎭挴
    #item鍜宺esult鍏跺疄鎸囧悜浜嗕竴涓猟ict瀵硅薄
    for item in trials.trials:
        trials_list.append(item)
    
    #鎸夌収鍏抽敭璇嶈繘琛屾帓搴忥紝鍏抽敭璇嶅嵆涓篿tem['result']['loss']
    trials_list.sort(key=lambda item: item["result"]["loss"])
    
    print("best parameter is:", trials_list[0])
    print()
    
def exist_files(title):
    
    return os.path.exists(title+"_best_model.pickle")
    
def save_inter_params(trials, space_nodes, best_nodes, title):
 
    files = open(str(title+"_intermediate_parameters.pickle"), "wb")
    pickle.dump([trials, space_nodes, best_nodes], files)
    files.close()

def load_inter_params(title):
  
    files = open(str(title+"_intermediate_parameters.pickle"), "rb")
    trials, space_nodes, best_nodes = pickle.load(files)
    files.close()
    
    return trials, space_nodes ,best_nodes
    
def save_best_model(best_model, title):
    
    files = open(str(title+"_best_model.pickle"), "wb")
    pickle.dump(best_model, files)
    files.close()
    
def load_best_model(title):
    
    files = open(str(title+"_best_model.pickle"), "rb")
    best_model = pickle.load(files)
    files.close()
    
    return best_model
    
def record_best_model_acc(clf, acc, best_model, best_acc):
    
    flag = False
    
    if not isclose(best_acc, acc):
        if best_acc < acc:
            flag = True
            best_acc = acc
            best_model = clf
            
    return best_model, best_acc, flag

def noise_augment_data(mean, std, X_train, Y_train, columns):
    
    X_noise_train = X_train.copy()
    X_noise_train.is_copy = False
    
    row = X_train.shape[0]
    for i in range(0, row):
        for j in columns:
            X_noise_train.iloc[i,[j]] +=  random.gauss(mean, std)

    return X_noise_train, Y_train
    
#鎴戣寰楄繖涓腑鏂囨枃妗ｄ粙缁峢yperopt杩樻槸姣旇緝濂絟ttps://www.jianshu.com/p/35eed1567463
def nn_f(params):
    
    print("mean", params["mean"])
    print("std", params["std"])
    print("lr", params["lr"])
    print("optimizer__weight_decay", params["optimizer__weight_decay"])
    print("criterion", params["criterion"])
    print("batch_size", params["batch_size"])
    print("optimizer__betas", params["optimizer__betas"])
    print("module", params["module"])    
    
    X_noise_train, Y_noise_train = noise_augment_data(params["mean"], params["std"], X_train_scaled, Y_train, columns=[3, 4, 5, 6, 7, 8])
    
    clf = NeuralNetClassifier(lr = params["lr"],
                              optimizer__weight_decay = params["optimizer__weight_decay"],
                              criterion = params["criterion"],
                              batch_size = params["batch_size"],
                              optimizer__betas = params["optimizer__betas"],
                              module=params["module"],
                              max_epochs = params["max_epochs"],
                              callbacks=[skorch.callbacks.EarlyStopping(patience=params["patience"])],
                              device = best_nodes["device"],
                              optimizer = best_nodes["optimizer"]
                              )
    
    skf = StratifiedKFold(Y_noise_train, n_folds=5, shuffle=True, random_state=None)
    
    clf.module.init_weight(params["init_mode"])
    
    metric = cross_val_score(clf, X_noise_train.values.astype(np.float32), Y_noise_train.values.astype(np.longlong), cv=skf, scoring="accuracy").mean()
    
    print(metric)
    print()    
    return -metric

def display_search_progress(search_times, nn_f):
    
    print("search times:", search_times)
    return nn_f
    
def parse_space(trials, space_nodes, best_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    best_nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
    best_nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
    best_nodes["max_epochs"] = space_nodes["max_epochs"][trials_list[0]["misc"]["vals"]["max_epochs"][0]]
    best_nodes["lr"] = trials_list[0]["misc"]["vals"]["lr"][0]
    best_nodes["module"] = space_nodes["module"][trials_list[0]["misc"]["vals"]["module"][0]] 
    best_nodes["optimizer__betas"] = space_nodes["optimizer__betas"][trials_list[0]["misc"]["vals"]["optimizer__betas"][0]]
    best_nodes["optimizer__weight_decay"] = space_nodes["optimizer__weight_decay"][trials_list[0]["misc"]["vals"]["optimizer__weight_decay"][0]]
    best_nodes["init_mode"] = space_nodes["init_mode"][trials_list[0]["misc"]["vals"]["init_mode"][0]]
    best_nodes["patience"] = space_nodes["patience"][trials_list[0]["misc"]["vals"]["patience"][0]]
    best_nodes["device"] = space_nodes["device"][trials_list[0]["misc"]["vals"]["device"][0]]
    best_nodes["optimizer"] = space_nodes["optimizer"][trials_list[0]["misc"]["vals"]["optimizer"][0]]
    
    return best_nodes
    
def predict(best_nodes, max_evals=10):
    
    best_acc = 0.0
    best_model = 0.0
    if (exist_files(best_nodes["title"])):
        best_model = load_best_model(best_nodes["title"])
        best_acc = cal_nnclf_acc(best_model, X_train_scaled, Y_train)
         
    for i in range(0, max_evals):
        
        print(str(i+1)+"/"+str(max_evals)+" prediction progress have been made.")
        
        clf = NeuralNetClassifier(lr = best_nodes["lr"],
                                  optimizer__weight_decay = best_nodes["optimizer__weight_decay"],
                                  criterion = best_nodes["criterion"],
                                  batch_size = best_nodes["batch_size"],
                                  optimizer__betas = best_nodes["optimizer__betas"],
                                  module=best_nodes["module"],
                                  max_epochs = best_nodes["max_epochs"],
                                  callbacks = [skorch.callbacks.EarlyStopping(patience=best_nodes["patience"])],
                                  device = best_nodes["device"],
                                  optimizer = best_nodes["optimizer"]
                                  )
        
        #鍦ㄨ繖閲嶆柊鍒濆鍖栦竴娆″熀鏈氨浼氬緱鍒板樊寮傚緢澶х殑缁撴灉鍚�
        #鐜板湪鍙互鐪嬪埌纭疄鏄粡杩囦簡鏉冮噸鍒濆鍖栨ā鍨嬮噸鏂拌缁冪殑
        #浣嗘槸杩欎釜best_model鐨勬暟鎹�庝箞杩樻槸涓婁竴鍥炵殑鏁版嵁锛�
        clf.module.init_weight(best_nodes["init_mode"])
        
        clf.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 
        
        metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
        print_nnclf_acc(metric)
        
        #閭ｄ箞鐜板湪鐨刡est_model搴旇灏变笉浼氳淇敼浜嗗惂
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
        #閫氳繃涓嬮潰鐨勪袱琛屼唬鐮佸彲浠ュ彂鐜癱lf纭疄姣忔閮芥槸鏂板垱寤虹殑锛屼絾鏄痬odule姣忔閮芥槸閲嶅浣跨敤鐨�
        #print(id(clf))
        #print(id(clf.module))
    
        if (flag):
            save_best_model(best_model, best_nodes["title"])
            Y_pred = best_model.predict(X_test_scaled.values.astype(np.float32))
            
            data = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
            output = pd.DataFrame(data = data)
            output.to_csv(best_nodes["path"], index=False)
            print("prediction file has been written.")
        print()
     
    #鍥犱负涓嬮潰鐨刢lf涓殑module宸茬粡琚噸鏂拌缁冧簡锛屾墍浠ュ凡缁忔槸鏂扮殑妯″瀷浜嗭紝杩樻槸鐩存帴杈撳嚭best_acc   
    #metric = cal_nnclf_acc(best_model, X_train_scaled, Y_train)
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    
#鎴戠湡鐨勬浌涔愶紝鍋氫笉鍋氭暟鎹泦澧炲己濂藉儚宸埆寰堝ぇ鍝︼紝涓嶆坊鍔犲櫔澹板噯纭巼楂樺緱澶氬憿銆傘�傚ソ鍍忎篃涓嶆槸
#涓婂洖閭ｄ釜璨屼技鏄阀鍚堣�屽凡锛屾垜澶氳繍琛屼簡鍑犳鍙戠幇鍔犱簡鍣０濂藉儚鏄楂樹竴鐐圭偣鍛€�傘��
#鍊掓槸patience璁剧疆涓�10鐨勬椂鍊欑粷澹佹病鏈夎缃负5鐨勬椂鍊欐晥鏋滃ソ鍛€�傘�備笉鐪嬭繍琛岃緭鍑虹湡杩樹笉鐭ラ亾
#鐜板湪鎴戠殑浠ｇ爜搴旂敤鍒颁笅涓�涓増鏈殑鏃跺�欏彧闇�瑕佷慨鏀箂pace銆乻pace_nodes浠ュ強best_nodes鍜宲arse_space
#predict鍑芥暟鍐卍ata = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
space = {"title":hp.choice("title", ["titanic"]),
         "path":hp.choice("path", ["C:/Users/1/Desktop/Titanic_Prediction.csv"]),
         "mean":hp.choice("mean", [0]),
         #"std":hp.choice("std", [0]),
         "std":hp.choice("std", [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]),
         "max_epochs":hp.choice("max_epochs",[400]),
         "patience":hp.choice("patience", [1,2,3,4,5,6,7,8,9,10]),
         "lr":hp.uniform("lr", 0.0001, 0.0015),  
         "optimizer__weight_decay":hp.choice("optimizer__weight_decay",
            [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
             0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019]),  
         "criterion":hp.choice("criterion", [torch.nn.NLLLoss, torch.nn.CrossEntropyLoss]),

         "batch_size":hp.choice("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
         "optimizer__betas":hp.choice("optimizer__betas",
                                      [[0.86, 0.9991], [0.86, 0.9993], [0.86, 0.9995], [0.86, 0.9997], [0.86, 0.9999],
                                       [0.88, 0.9991], [0.88, 0.9993], [0.88, 0.9995], [0.88, 0.9997], [0.88, 0.9999],
                                       [0.90, 0.9991], [0.90, 0.9993], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999],
                                       [0.92, 0.9991], [0.92, 0.9993], [0.92, 0.9995], [0.92, 0.9997], [0.92, 0.9999],
                                       [0.94, 0.9991], [0.94, 0.9993], [0.94, 0.9995], [0.94, 0.9997], [0.94, 0.9999]]),
         "module":hp.choice("module", [module1, module2, module3, module4, module5, module6, module7, module8,
                                       module9, module10, module11, module12, module13, module14, module15, module16,
                                       module17, module18, module19, module20, module21, module22, module23, module24]),
         "init_mode":hp.choice("init_mode", [1, 2, 3, 4]),         
         "device":hp.choice("device", ["cpu"]),
         "optimizer":hp.choice("optimizer", [torch.optim.Adam])
         }

space_nodes = {"title":["titanic"],
               "path":["path"],
               "mean":[0],
               #"std":[0],
               "std":[0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
               "max_epochs":[400],
               "patience":[1,2,3,4,5,6,7,8,9,10],
               "lr":[0.0001],
               "optimizer__weight_decay":[0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                                          0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019],
               "criterion":[torch.nn.NLLLoss, torch.nn.CrossEntropyLoss],
               "batch_size":[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
               "optimizer__betas":[[0.86, 0.9991], [0.86, 0.9993], [0.86, 0.9995], [0.86, 0.9997], [0.86, 0.9999],
                                   [0.88, 0.9991], [0.88, 0.9993], [0.88, 0.9995], [0.88, 0.9997], [0.88, 0.9999],
                                   [0.90, 0.9991], [0.90, 0.9993], [0.90, 0.9995], [0.90, 0.9997], [0.90, 0.9999],
                                   [0.92, 0.9991], [0.92, 0.9993], [0.92, 0.9995], [0.92, 0.9997], [0.92, 0.9999],
                                   [0.94, 0.9991], [0.94, 0.9993], [0.94, 0.9995], [0.94, 0.9997], [0.94, 0.9999]],
               "module":[module1, module2, module3, module4, module5, module6, module7, module8,
                         module9, module10, module11, module12, module13, module14, module15, module16,
                         module17, module18, module19, module20, module21, module22, module23, module24],
               "init_mode":[1, 2, 3, 4],
               "device":["cpu"],
               "optimizer":[torch.optim.Adam]
               }

best_nodes = {"title":"titanic",
              "path":"path",
              "mean":0,
              "std":0.1,
              "max_epochs":400,
              "patience":5,
              "lr":0.0001,
              "optimizer__weight_decay":0.005,
              "criterion":torch.nn.NLLLoss,
              "batch_size":1,
              "optimizer__betas":[0.86, 0.999],
              "module":module3,
              "init_mode":1,
              "device":"cpu",
              "optimizer":torch.optim.Adam
              }

#鎴戣寰楄繖杈归渶瑕佹坊鍔犱竴涓绠楄鏃剁殑鍔熻兘
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

best_params = fmin(nn_f, space, algo=algo, max_evals=1, trials=trials)
print_best_params_acc(trials)

best_nodes = parse_space(trials, space_nodes, best_nodes)
#save_inter_params淇濆瓨鐨勬槸鏈鎼滅储鍒扮殑鍙傛暟
save_inter_params(trials, space_nodes, best_nodes, "titanic")
trials, space_nodes, best_nodes = load_inter_params("titanic")

#predict涓殑best_model淇濆瓨鐨勬槸鏈満杩愯杩囩▼涓渶浣虫ā鍨�
#鐜板湪杩樻湁涓�涓鎬殑闂锛宲redict涓技涔庤繕鏄笉瀵癸紝鍥犱负鍒濆姝ｇ‘鐜囧お楂樹簡鍚�
#缁忚繃鎴戞祴璇曪紝鎴戝彂鐜皃redict涓垵濮嬫纭巼纭疄鏈夊湪鍙樺寲锛屾墍浠ュ簲璇ユ湪闂鍚�
#鐜板湪灏辨槸灏嗘墍鏈夊鏄撴敼鍙樼殑涓滆タ閮芥斁鍒皊pace銆乻pace_nodes銆乥est_nodes浠ュ強parse_space
#杩欐牱鐨勫仛娉曟湁鍒╀簬閬垮厤鎴戝繕璁颁慨鏀逛唬鐮佺粏鑺備粠鑰屽鑷村奖鍝嶆ā鍨嬬殑璁粌杩囩▼
#鎴戞劅瑙夐櫎浜嗗拰妯″瀷鐩稿叧鐨勮秴鍙傛垜宸茬粡鎼炲畾鐨勫樊涓嶅浜嗭紝浠婂悗涓昏鍐崇瓥鍜屾ā鍨嬬浉鍏崇殑瓒呭弬
#姣斿璇存槸妯″瀷鐨勫眰鏁般�佹瘡灞傜殑鑺傜偣鏁般�佸垵濮嬪寲鐨勬柟寮忋�佸垵濮嬪寲鐨勮寖鍥淬�佸亸缃殑璁剧疆鍊�
#鏄庡ぉ鐨勫伐浣滃氨鍏堜粠妯″瀷鐢熸垚鍣ㄥ紑濮嬪挴銆傘��
predict(best_nodes, max_evals=10)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

print("mother fucker~")
"""
#妯″瀷涓�鏃﹁淇敼浜嗕箣鍚巔ickle涓殑鏂囦欢鍐嶈鍑烘潵灏辨病鐢ㄥ暒
#杩樺ソ鎴戝強鏃朵繚瀛樹簡鑰佺増鏈殑妯″瀷锛屼笉鐒舵劅瑙夊氨蹇冮吀浜嗐�傘�傘��
#涓嶈繃鍏跺疄娌′繚瀛橀棶棰樹篃涓嶅ぇ鐨勫惂锛岀洿鎺ラ噸鏂拌绠椾竴娆°�傘�傘��
#鍚屼竴鏃ユ湡涓嬬殑涓棿缁撴灉鍜屾渶浣虫ā鍨嬫槸閰嶅鐨勶紝铏界劧杩欎腑闂寸粨鏋滄湭蹇呬骇鐢熶簡鏈�浣虫ā鍨�
files = open("titanic_intermediate_parameters_2018-9-15221036.pickle", "rb")
trials, space_nodes, best_nodes = pickle.load(files)
files.close()
print(best_nodes)
#print(space_nodes)

files = open("titanic_best_model_2018-9-15221021.pickle", "rb")
best_model = pickle.load(files)
files.close()
best_acc = cal_nnclf_acc(best_model, X_train_scaled, Y_train)
print(best_acc)
"""