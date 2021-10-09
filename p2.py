import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt  
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('Email_logistic.csv', header=0)
feature = ['image', 'attach', 'dollar','inherit', 'viagra', 'password']
x = data[feature]
y = data.spam
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.2 ,random_state=40)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
score = logreg.score(X_test,y_test)
print(score)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

lsvc = LinearSVC(C=100, random_state=10, tol=1e-4)
lsvc.fit(X_train, y_train)
print(lsvc.score(X_train, y_train))
ypred2 = lsvc.predict(X_test)
cnf_matrix2 = metrics.confusion_matrix(y_test, ypred2)
print(cnf_matrix2)

knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
ypred3 = knn.predict(X_test)
score = knn.score(X_test,y_test)
print(score)

cnf_matrix3 = metrics.confusion_matrix(y_test, ypred3)
print(cnf_matrix3)

axis = ['dollar', 'attach', 'password']
xVals = data[axis]
yVals = data.spam

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
colormap = np.array(['b', 'r'])
ax.scatter(xVals['dollar'], xVals['attach'], xVals['password'], c = colormap[yVals], marker = 'o')

ax.set_xlabel('Dollars')
ax.set_ylabel('Attach')
ax.set_zlabel('Password')

plt.show()
