
import numpy as np
from RandomForest import RandomForest
from sklearn import datasets
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()

X, y = bc.data , bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

clf = RandomForest()
clf.fit(X_train, y_train)

pridictions = clf.predict(X_test)

accuracy = np.sum(pridictions==y_test)/len(y_test)
print(accuracy)
