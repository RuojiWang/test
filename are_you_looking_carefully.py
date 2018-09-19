from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
#����߼��ع�ȷʵ�ǲ�ֹһ��epoch��������max_iter�������Կ�����
#��������ĵ����������ĵ���˵��max_iter��ʱ��Ҳ�ᵽ��epochs
#�����������Ѿ��ҵ��˴𰸣��߼��ع�ȷʵѧϰһ��epoch
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
lr.fit(X_train_std, y_train)
print(lr.score(X_train_std, y_train))