"""
Created on Thu Mar 26 00:33:13 2020

@author: raghav
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('tin_binary_labelled.csv')
#dataset = pd.read_csv('tinlet_labelled_pepsico.csv')

""" x = process_variable_values """
x = dataset.iloc[:, 0].values

""" y = states """
y = dataset.iloc[:, 1].values

""" Split dataset into train and test sets """
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

""" Convert to 1D array """
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

""" Feature Scaling """
scaler_object = StandardScaler()
x_train = scaler_object.fit_transform(x_train) 
x_test = scaler_object.transform(x_test)

""" Classify train set to learn the correlation between variable_values and states """
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

""" Predict states from values """
predicted_states = classifier.predict(x_test)

""" Generate Confusion Matrix """
confusionMatrix = confusion_matrix(y_test, predicted_states)

"""" Accuracy of predicted values with test set """
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predicted_states)

cm = confusion_matrix(y_test, predicted_states)

#conf_matrix = [[2867, 6], [0, 13801]]

import numpy as np
#conf_matrix = np.array(conf_matrix)
import matplotlib.pyplot as plt
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Running State', 'Downtime State']
#plt.title('Logistic Regression - Tin - Confusion Matrix')
plt.ylabel('Actual Data')
plt.xlabel('Predicted Data')
#plt.axis('off')
tick_marks = np.arange(len(classNames))
plt.yticks(tick_marks, classNames)
plt.xticks(tick_marks, classNames)
#s = [['TP','FN'], ['FP', 'TN']]

for i in range(2):
    for j in range(2):
        plt.text( j,i,str(cm[i][j]), horizontalalignment = "center")
plt.show()

print("\nAccuracy = {0}%".format(accuracy*100))

zerocount = 0
onecount = 0
for i in range(0, len(predicted_states)):
    if predicted_states[i] == 0:
        zerocount+=1
    else:
        onecount+=1
        
print("\nNo of Running States - {0}".format(zerocount))
print("\nNo of Downtime States - {}".format(onecount))
#print(zerocount, onecount)