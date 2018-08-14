import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("D:\KVCH\Prostate Cancer\Prostate Cancer Case Study KNN\Prostate_Cancer.csv")
print(len(dataset))
print(dataset.head())

#--Replace zeroes
zero_not_accepted = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

    
#--Split dataset
X = dataset.iloc[:, 2:10]
Y = dataset.iloc[:, 1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=42)


#--Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



#--Define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')
knn = KNeighborsClassifier()
classifier.fit(X_train, Y_train)
#--predict the test set results
Y_pred = classifier.predict(X_test)
#print(Y_pred)



#print('Accuracy of KNN on training set: {:.3f}'.format(knn.score(X_train, Y_train)))
#print('Accuracy of KNN on testing set: {:.3f}'.format(knn.score(X_test, Y_test)))


#--Evaluate Model
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print(accuracy_score(Y_test, Y_pred))



#-- test the algorithm using range of neighbors 2-9, to see which no of neighbor
# yeilds the best result in terms of production
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=42)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(2,10)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, Y_train)
    training_accuracy.append(clf.score(X_train, Y_train))
    test_accuracy.append(clf.score(X_test, Y_test))
    
plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the test set')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbor')
plt.legend()
  
