# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:21:01 2021

@author: Dietrich
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split



#%% Data import and short inspection

df = pd.read_csv('../data/aug_train.csv')
row_count = df.shape[0]

# Inspect NaN values
print(df.isna().sum())
print((df.isna().sum()/row_count)*100)

# Check correlation of each column with target column
numeric_features = df.drop(['enrollee_id', 'city'], axis=1)
numeric_features = pd.get_dummies(numeric_features)
corr = numeric_features.corr()
top_corr = corr['target'].sort_values(ascending=False)[:6]
print(corr['target'].sort_values(ascending=False)[:10],'\n')
print(corr['target'].sort_values(ascending=False)[-10:])


#%% Random Oversampling

##### From https://www.machinecurve.com/index.php/2020/11/10/working-with-imbalanced-datasets-with-tensorflow-and-keras/

# Count samples per class
classes_zero = df[df['target'] == 0]
classes_one = df[df['target'] == 1]

# Print sizes
print(f'Class 0: {len(classes_zero)}')
print(f'Class 1: {len(classes_one)}')

# Oversample one to the size of zero
classes_one = classes_one.sample(len(classes_zero), replace=True)

# Print sizes
print(f'Class 0: {len(classes_zero)}')
print(f'Class 1: {len(classes_one)}')

df = df.sample(len(classes_zero), replace=True)

#%% Model

X = df.drop(['enrollee_id', 'city', 'target'], axis=1) #'gender', 'company_size', 'company_type', 'major_discipline', 
X = pd.get_dummies(X)
Y = df.target

# Splitting into Training/Testing/Validation
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3,
                                                    random_state=13)
X_test, X_valid, y_test, y_valid = train_test_split(X_valid, y_valid, test_size=0.5,
                                                    random_state=13)


num_inputs = np.shape(X)[1]

batch_size = 50


# Create and fit model
def create_model():
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(128, input_dim=num_inputs, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return model

model = create_model()
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=batch_size)


# Plot Model Accuracy and Loss
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy vs Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate model with test data and print scores
scores = model.evaluate(X_test, y_test) # batch_size=batch_size)
print('Scores on testing data:', scores)
#%% Grid Search CV

# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# model = KerasClassifier(build_fn=create_model,verbose=2)
# batch_size = [20, 40, 60]
# epochs = [5, 10]
# param_grid = dict(batch_size = batch_size, epochs = epochs)
# grid = GridSearchCV(estimator=model, param_grid = param_grid)
# grid_result = grid.fit(X_train, y_train)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))