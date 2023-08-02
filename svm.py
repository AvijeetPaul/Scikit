from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

#split it in features and lables
X= iris.data
y=iris.target

classes=["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

print("The shape of X is " , X.shape)
print("The shape of y is " ,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# print()
# print("The shape of X_train is " ,X_train.shape)
# print("The shape of X_test is " ,X_test.shape)
# print("The shape of y_train is " ,y_train.shape)
# print("The shape of y_test is " ,y_test.shape)

model= svm.SVC()
model.fit(X_train,y_train)

print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test,predictions)

print("Prediction: ", predictions)
print("actual: ",y_test)
print("Accuracy: ", acc)

for i in range(len(predictions)):
    print(classes[predictions[i]])