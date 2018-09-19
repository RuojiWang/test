#coding=utf-8

print("mother fucker")

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
    
#鏉╂瑩鍣烽惃鍒磑de閺勵垱鐪扮憴顤禷ndas.core.series.Series娴兼鏆熼惃鍕儑娑擄拷娑擃亜锟界》绱欓崣顖濆厴閺堝顦挎稉顏冪船閺佸府绱�
freq_port = data_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#鐏忓摼ata_test娑擃厾娈慺are閸忓啰绀岄幍锟界紓鍝勩亼閻ㄥ嫰鍎撮崚鍡欐暠瀹歌尙绮￠崠鍛儓閻ㄥ嫭鏆熼幑顔炬畱娑擃厺缍呴弫鏉垮枀鐎规艾鎼�
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

#鐏忚偐甯ョ紒娆庣稑鐠囧娈戞潻娆庨嚋閺勵垵纭�閻氼喛鍩炵粊顭掔礉閸樼喐娼甸惃鍕閺傚洭鍣烽棃銏＄壌閺堫剙姘ㄥ▽鈩冩箒鏉╂瑧顫掔拠瀛樼《閸拷
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
#print(df)
df_ticket = df.index.values          #閸忓彉闊╅懜鍦偍閻ㄥ嫮銈ㄩ崣锟�
tickets = data_train.Ticket.values   #閹碉拷閺堝娈戦懜鍦偍
#print(tickets)
result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #闁秴宸婚幍锟介張澶庡煘缁侇煉绱濋崷銊ュ彙娴滎偉鍩炵粊銊╁櫡闂堛垻娈戞稉锟�1閿涘苯鎯侀崚娆庤礋0
    result.append(ticket)
    
df = data_train['Ticket'].value_counts()
df = pd.DataFrame(df)
df = df[df['Ticket'] > 1]
df_ticket = df.index.values          #閸忓彉闊╅懜鍦偍閻ㄥ嫮銈ㄩ崣锟�
tickets = data_train.Ticket.values   #閹碉拷閺堝娈戦懜鍦偍

result = []
for ticket in tickets:
    if ticket in df_ticket:
        ticket = 1
    else:
        ticket = 0                   #闁秴宸婚幍锟介張澶庡煘缁侇煉绱濋崷銊ュ彙娴滎偉鍩炵粊銊╁櫡闂堛垻娈戞稉锟�1閿涘苯鎯侀崚娆庤礋0
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
#閹存垼顫庡妤勵唲缂佸啴娉﹂崪灞剧ゴ鐠囨洟娉﹂棁锟界憰浣告躬娑擄拷鐠х柉绻樼悰宀�澹掑浣虹級閺�鎾呯礉閹碉拷娴犮儲鏁為柌濠冨竴娴滃棗甯弶銉ф畱X_train閻ㄥ嫮澹掑浣虹級閺�鎯ф尨
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告担璺ㄦ暏姒涙顓荤拋鍓х枂閻ㄥ嫭鍓伴幀婵嗘尨
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
            pass#鐏忚鲸妲告禒锟芥稊鍫ュ厴娑撳秴浠涢惃鍕壈閹拷
        
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
    #娴犲窐rials娑擃叀顕伴崣鏍ㄦ付婢堆呮畱閸戝棛鈥橀悳鍥︿繆閹垰鎸�
    #item閸滃esult閸忚泛鐤勯幐鍥ф倻娴滃棔绔存稉鐚焛ct鐎电钖�
    for item in trials.trials:
        trials_list.append(item)
    
    #閹稿鍙庨崗鎶芥暛鐠囧秷绻樼悰灞惧笓鎼村骏绱濋崗鎶芥暛鐠囧秴宓嗘稉绡縯em['result']['loss']
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
    
#閹存垼顫庡妤勭箹娑擃亙鑵戦弬鍥ㄦ瀮濡楋絼绮欑紒宄peropt鏉╂ɑ妲稿В鏃囩窛婵傜禑ttps://www.jianshu.com/p/35eed1567463
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
        
        #閸︺劏绻栭柌宥嗘煀閸掓繂顫愰崠鏍︾濞嗏�崇唨閺堫剙姘ㄦ导姘繁閸掓澘妯婂鍌氱发婢堆呮畱缂佹挻鐏夐崥锟�
        #閻滄澘婀崣顖欎簰閻鍩岀涵顔肩杽閺勵垳绮℃潻鍥︾啊閺夊啴鍣搁崚婵嗩潗閸栨牗膩閸ㄥ鍣搁弬鎷岊唲缂佸啰娈�
        #娴ｅ棙妲告潻娆庨嚋best_model閻ㄥ嫭鏆熼幑顔斤拷搴濈疄鏉╂ɑ妲告稉濠佺閸ョ偟娈戦弫鐗堝祦閿涳拷
        clf.module.init_weight(best_nodes["init_mode"])
        
        clf.fit(X_train_scaled.values.astype(np.float32), Y_train.values.astype(np.longlong)) 
        
        metric = cal_nnclf_acc(clf, X_train_scaled, Y_train)
        print_nnclf_acc(metric)
        
        #闁絼绠為悳鏉挎躬閻ㄥ垺est_model鎼存棁顕氱亸鍙樼瑝娴兼俺顫︽穱顔芥暭娴滃棗鎯�
        best_model, best_acc, flag = record_best_model_acc(clf, metric, best_model, best_acc)
        #闁俺绻冩稉瀣桨閻ㄥ嫪琚辩悰灞煎敩閻礁褰叉禒銉ュ絺閻滅櫛lf绾喖鐤勫В蹇旑偧闁姤妲搁弬鏉垮灡瀵よ櫣娈戦敍灞肩稻閺勭棳odule濮ｅ繑顐奸柈鑺ユЦ闁插秴顦叉担璺ㄦ暏閻拷
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
     
    #閸ョ姳璐熸稉瀣桨閻ㄥ垻lf娑擃厾娈憁odule瀹歌尙绮＄悮顐﹀櫢閺傛媽顔勭紒鍐х啊閿涘本澧嶆禒銉ュ嚒缂佸繑妲搁弬鎵畱濡�崇�锋禍鍡礉鏉╂ɑ妲搁惄瀛樺复鏉堟挸鍤璪est_acc   
    #metric = cal_nnclf_acc(best_model, X_train_scaled, Y_train)
    print("the best accuracy rate of the model on the whole train dataset is:", best_acc)
    
#閹存垹婀￠惃鍕祵娑旀劧绱濋崑姘瑝閸嬫碍鏆熼幑顕�娉︽晶鐐插繁婵傝棄鍎氬顔煎焼瀵板牆銇囬崫锔肩礉娑撳秵鍧婇崝鐘叉珨婢规澘鍣涵顔惧芳妤傛ê绶辨径姘喛閵嗗倶锟藉倸銈介崓蹇庣瘍娑撳秵妲�
#娑撳﹤娲栭柇锝勯嚋鐠ㄥ奔鎶�閺勵垰闃�閸氬牐锟藉苯鍑￠敍灞惧灉婢舵俺绻嶇悰灞肩啊閸戠姵顐奸崣鎴犲箛閸旂姳绨￠崳顏勶紣婵傝棄鍎氶弰顖濐洣妤傛ü绔撮悙鍦仯閸涒偓锟藉倶锟斤拷
#閸婃帗妲竝atience鐠佸墽鐤嗘稉锟�10閻ㄥ嫭妞傞崐娆戠卜婢逛焦鐥呴張澶庮啎缂冾喕璐�5閻ㄥ嫭妞傞崐娆愭櫏閺嬫粌銈介崨鈧拷鍌橈拷鍌欑瑝閻绻嶇悰宀冪翻閸戣櫣婀℃潻妯圭瑝閻儵浜�
#閻滄澘婀幋鎴犳畱娴狅絿鐖滄惔鏃傛暏閸掗绗呮稉锟芥稉顏嗗閺堫剛娈戦弮璺猴拷娆忓涧闂囷拷鐟曚椒鎱ㄩ弨绠俻ace閵嗕够pace_nodes娴犮儱寮穊est_nodes閸滃arse_space
#predict閸戣姤鏆熼崘鍗峚ta = {"PassengerId":data_test["PassengerId"], "Survived":Y_pred}
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

#閹存垼顫庡妤勭箹鏉堝綊娓剁憰浣瑰潑閸旂姳绔存稉顏囶吀缁犳顓搁弮鍓佹畱閸旂喕鍏�
start_time = datetime.datetime.now()

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)

best_params = fmin(nn_f, space, algo=algo, max_evals=1, trials=trials)
print_best_params_acc(trials)

best_nodes = parse_space(trials, space_nodes, best_nodes)
#save_inter_params娣囨繂鐡ㄩ惃鍕Ц閺堫剚顐奸幖婊呭偍閸掓壆娈戦崣鍌涙殶
save_inter_params(trials, space_nodes, best_nodes, "titanic")
trials, space_nodes, best_nodes = load_inter_params("titanic")

#predict娑擃厾娈慴est_model娣囨繂鐡ㄩ惃鍕Ц閺堫剚婧�鏉╂劘顢戞潻鍥┾柤娑擃厽娓舵担铏侀崹锟�
#閻滄澘婀潻妯绘箒娑擄拷娑擃亜顨岄幀顏嗘畱闂傤噣顣介敍瀹瞨edict娑擃厺鎶�娑斿氦绻曢弰顖欑瑝鐎电櫢绱濋崶鐘辫礋閸掓繂顫愬锝団�橀悳鍥с亰妤傛ü绨￠崥锟�
#缂佸繗绻冮幋鎴炵ゴ鐠囨洩绱濋幋鎴濆絺閻滅殐redict娑擃厼鍨垫慨瀣劀绾喚宸肩涵顔肩杽閺堝婀崣妯哄閿涘本澧嶆禒銉ョ安鐠囥儲婀梻顕�顣介崥锟�
#閻滄澘婀亸杈ㄦЦ鐏忓棙澧嶉張澶婎啇閺勬挻鏁奸崣妯兼畱娑撴粏銈块柈鑺ユ杹閸掔殜pace閵嗕够pace_nodes閵嗕攻est_nodes娴犮儱寮穚arse_space
#鏉╂瑦鐗遍惃鍕粵濞夋洘婀侀崚鈺�绨柆鍨帳閹存垵绻曠拋棰佹叏閺�閫涘敩閻胶绮忛懞鍌欑矤閼板苯顕遍懛鏉戝閸濆秵膩閸ㄥ娈戠拋顓犵矊鏉╁洨鈻�
#閹存垶鍔呯憴澶愭珟娴滃棗鎷板Ο鈥崇�烽惄绋垮彠閻ㄥ嫯绉撮崣鍌涘灉瀹歌尙绮￠幖鐐茬暰閻ㄥ嫬妯婃稉宥咁樋娴滃棴绱濇禒濠傛倵娑撴槒顩﹂崘宕囩摜閸滃本膩閸ㄥ娴夐崗宕囨畱鐡掑懎寮�
#濮ｆ柨顩х拠瀛樻Ц濡�崇�烽惃鍕湴閺佽埇锟戒焦鐦＄仦鍌滄畱閼哄倻鍋ｉ弫鑸拷浣稿灥婵瀵查惃鍕煙瀵繈锟戒礁鍨垫慨瀣閻ㄥ嫯瀵栭崶娣拷浣镐焊缂冾喚娈戠拋鍓х枂閸婏拷
#閺勫骸銇夐惃鍕紣娴ｆ粌姘ㄩ崗鍫滅矤濡�崇�烽悽鐔稿灇閸ｃ劌绱戞慨瀣尨閵嗗倶锟斤拷
predict(best_nodes, max_evals=10)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))

print("mother fucker~")
"""
#濡�崇�锋稉锟介弮锕侇潶娣囶喗鏁兼禍鍡曠閸氬窋ickle娑擃厾娈戦弬鍥︽閸愬秷顕伴崙鐑樻降鐏忚鲸鐥呴悽銊ユ殥
#鏉╂ê銈介幋鎴濆挤閺冩湹绻氱�涙ü绨￠懓浣哄閺堫剛娈戝Ο鈥崇�烽敍灞肩瑝閻掕埖鍔呯憴澶婃皑韫囧啴鍚�娴滃棎锟藉倶锟藉倶锟斤拷
#娑撳秷绻冮崗璺虹杽濞屸�茬箽鐎涙﹢妫舵０妯圭瘍娑撳秴銇囬惃鍕儌閿涘瞼娲块幒銉╁櫢閺傛媽顓哥粻妞剧濞喡帮拷鍌橈拷鍌橈拷锟�
#閸氬奔绔撮弮銉︽埂娑撳娈戞稉顓㈡？缂佹挻鐏夐崪灞炬付娴ｈ櫕膩閸ㄥ妲搁柊宥咁殰閻ㄥ嫸绱濋搹鐣屽姧鏉╂瑤鑵戦梻瀵哥波閺嬫粍婀箛鍛獓閻㈢喍绨￠張锟芥担铏侀崹锟�
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