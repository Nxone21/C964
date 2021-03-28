import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import pickle

# Gather the data
wine_df = pd.read_csv('Downloads/winequality-red.csv')
print("Dataset Shape:", wine_df.shape)

# checking the data
wine_df.head(20)

# clean and preprocess the data
clean_wine = wine_df.drop(['fixed acidity', 'residual sugar', 'free sulfur dioxide',
                           'total sulfur dioxide', 'density'], axis=1)

# check cleaned up data
clean_wine.head(20)

# create classification of target variables
bins = [0, 5.5, 10]  # means 0-6 is okay wine, 6-10 is great wine.
labels = [0, 1]
clean_wine['quality'] = pd.cut(clean_wine['quality'], bins=bins, labels=labels)
clean_wine.head(20)

# normalize feature variables
x = clean_wine[clean_wine.columns[:-1]]
y = clean_wine['quality']
# sc = StandardScaler()
# x = sc.fit_transform(x)

# this splits the data to training data and testing data.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=42)

# Random forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)

# print accuracy and also total, correct and incorrect predictions
print("Total predictions: ", len(pred_rf))
print("correct predictions: ", sum(pred_rf == y_test))
print("incorrect predictions: ", sum(pred_rf != y_test))
print("accuracy: ", sum(pred_rf == y_test) * 100 / len(pred_rf), '%')

# confusion matrix using random forest
cm = confusion_matrix(y_test, pred_rf)
print(cm)

# matplot for confusion matrix
fig = plt.figure(figsize=(10, 6))
plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show

# dumping the model to the disk
pickle.dump(rf, open('wine_model.pkl', 'wb'))

# loading the model from disk
pred_model = pickle.load(open('wine_model.pkl', 'rb'))
