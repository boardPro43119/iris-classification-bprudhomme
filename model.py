# Importing the libraries
import numpy as np
import pandas as pd
import pickle

from sklearn import datasets
iris = datasets.load_iris()

# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# df["target"] = iris.target

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier()

classifier.fit(x_train,y_train)

pickle.dump(classifier, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))