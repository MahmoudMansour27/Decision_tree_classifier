# Decision Tree Iris Classifier

# import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 5:].values

# Encoding categorical data
labelencoder_y = LabelEncoder()
y[:, 0] = labelencoder_y.fit_transform(y[:, 0])

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train decision tree classifier model
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train.astype('int'))

# predict testing set
y_pred = classifier.predict(X_test)


# confusion matrix
cm = confusion_matrix(y_test.astype('int'), y_pred.astype('int'))

# Visualising Decision Tree
fig = plt.figure(figsize=(15,10))
tree.plot_tree(classifier, filled=True, rounded=True)

#viz = dt.dtreeviz(classifier, X_train, y_train, target_name='target')



