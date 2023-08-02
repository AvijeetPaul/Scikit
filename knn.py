import numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df= pd.read_csv("car.data")
#print(df.head())

X=df[["buying","maint","safety"]].values
y= df[["class"]]

#print(X,y)

#converting the data
Le=LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = Le.fit_transform(X[:,i])

#print(X,y)

#converting the label
label_mapping={
    "unacc":0,
    "acc":1,
    "good":2,
    "vgood":3
               }
y["class"] = y["class"].map(label_mapping)
y=np.array(y)

#print(y)

#create model
knn= neighbors.KNeighborsClassifier(n_neighbors=25, weights="uniform")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

knn.fit(X_train,y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test,prediction)

print("Prediction: ",prediction)
print("Accuracy: ", accuracy)

a=172
print("actual value: ",y[a])
print("predicted value: ",knn.predict(X)[a])