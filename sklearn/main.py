from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#import  numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
Y = iris.target
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) #随机数种子random_state=A 使每次随机相同
print(y_train)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
#print(x_train_std)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(x_train_std, y_train)
y_pred = ppn.predict(x_test_std)
print('misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(x_train_std, y_train)
re = lr.predict_proba(x_test_std[[0], :])
print(re)