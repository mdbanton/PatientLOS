# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:00:27 2021

@author: mlove
"""


import sklearn
import pandas as pd
import numpy as np
#import tensorflow as tf

from sklearn import svm

from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import TomekLinks
#from imblearn.over_sampling import ADASYN


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import History 

from tensorflow.keras.models import Sequential

#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

#from tensorflow.keras import Model
#from tensorflow.keras.layers import Conv2D
#from tensorflow.keras.layers.convolutional import MaxPooling2D
#from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
#from tensorflow.keras.layers import Reshape
#from tensorflow.keras.layers import BatchNormalization

#from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.metrics import PrecisionAtRecall


import warnings
warnings.filterwarnings('ignore')

def create_model(learn_rate=0.01, dropout_rate=0.4
                 ):
	# create model
    model = Sequential()
    model.add(Dense(400, activation='softmax', input_dim=int(inputShape)))

    model.add(Dense(200))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
                 
    model.add(Dense(4, activation='softmax'))
    optimizer = Adam(learning_rate=learn_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy', 'accuracy']
                 )

    return model

class Histories(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.aucs = []
        self.losses = []
        return

    def on_epoch_end(self, epoch, logs={}):
        self.aucs = []
        self.losses = []
        print("Epoch ended")
        return

    def on_batch_begin(self, batch, logs={}):
        self.aucs = []
        self.losses = []
        return

    def on_batch_end(self, batch, logs={}):
        return


df = pd.read_csv("Hospital_Inpatient_Discharges__SPARCS_De-Identified___2017.csv",
                 dtype={'Birth Weight': 'string'}
                 )



df = pd.DataFrame(df, columns= ['Age Group', 'Gender', 'Race', 'Ethnicity', 
                                'Type of Admission', 'Patient Disposition', 
                                'CCS Diagnosis Description', 'CCS Procedure Code', 
                                'CCS Procedure Description', 'APR DRG Code', 
                                'APR MDC Code', 'APR Severity of Illness Code', 
                                'APR Risk of Mortality', 
                                'Emergency Department Indicator', 'Length of Stay'
                                ])

# Uncomment to get the head
#pd.set_option('max_columns', df.columns.size)
#print(df.head(5))

df = pd.DataFrame(df.dropna())
dy = pd.DataFrame(df, columns= ['Length of Stay'])

listy = []
d = dict()

# Uncomment to get list of CCS Diagnosis Description, or change to get values 
#for any other feature
# dAd = dict()
# for index, row in df.iterrows():
#     if row['CCS Diagnosis Description'] in dAd:
#         dAd[row['CCS Diagnosis Description']] = dAd[row['CCS Diagnosis Description']] + 1
#     else:
#         dAd[row['CCS Diagnosis Description']] = 1

# print(dAd)

# Set Y values
count = 0
for index, row in dy.iterrows():
    count = count + 1
    if row['Length of Stay'].isdigit():
       if 0 <= int(row['Length of Stay']) <= 1:
           listy.append(1)
       elif int(row['Length of Stay']) == 2:
           listy.append(1)
       elif int(row['Length of Stay']) == 3:
           listy.append(2)
       elif 4 <= int(row['Length of Stay']) <= 5:
           listy.append(2)
       elif 6 <= int(row['Length of Stay']) <= 9:
           listy.append(2)
       elif 10 <= int(row['Length of Stay']) <= 120:
           listy.append(3)
    else:
        if row['Length of Stay'] == '120 +':
            listy.append(3)
        else:
            #This should not appear
            print('We got something weird here!!')
print("Count is: ", count)

# Uncomment to get counts for patient LOS
# for row in listy:
#     if row in d:
#         d[row] = d[row] + 1
#     else:
#         d[row] = 1

# Finish setting up the dataset
Y = np.array(listy)
df = df.drop(columns = 'Length of Stay')
Y = Y.reshape(-1, 1)

# Ensure Y is shaped correctly
print("Y is shape ", Y.shape)


# Set encoder and one Hot encode X
enc = OneHotEncoder()

enc.fit(df)
transformed = enc.transform(df)

# Ensure X is shaped correctly
print("X is shape ", transformed.shape)
X = transformed

# The dataset is too large to use as is, get a subset of data for use as 
# training, testing and evaluating. Evaluating used for NN only, as test data
# is used as part of training to validate the model. 
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, 
                                                    test_size=0.05, 
                                                    train_size=0.05)
X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, 
                                                  test_size=0.5)


# The feature space has been increased through One Hot Encoding, this has also
# led to irrelevant features being introduced, reduce it with PCA
print('PCA Starting')
# Reasonable value cosen, testing could be made to tune it further
pca = PCA(0.98)
pca.fit(X_train.toarray())

# SMOTE requires toarray()
X_train = pca.transform(X_train.toarray())
X_test = pca.transform(X_test.toarray())
X_eval = pca.transform(X_eval.toarray())

# This can take a while
print("Begin SMOTE")
smo = SMOTE()
X_train, y_train = smo.fit_resample(X_train, y_train)
print("End SMOTE")

# Some models require categorical values for Y, change is made here
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)
Y_eval = to_categorical(y_eval)

# One last check shape is ok
print("X is shape ", X_train.shape)
print("Y is shape ", Y_train.shape)

#ydict = dict()

###############################################################
###############################################################
#print(y_train)
#print(y_test)

# for row in y_test:
# #    print(int(row))
#     if int(row) in ydict:
#         ydict[int(row)] = ydict[int(row)] + 1
#     else:
#         ydict[int(row)] = 1

# for key in list(ydict.keys()):
#     print(key, ":", ydict[key])

# print("y_test has: ", ydict)

# Set the number of classes we're working with
num_classes = 3

print("*************************************")
print("*************************************")
print("************NEURAL NETWORK************")
print("*************************************")
print("*************************************")


print("Neural network")

model = KerasClassifier(build_fn=create_model, verbose=50, epochs=500)
# define the grid search parameters
inputShape = int(X_train.shape[1])
batch_size = [300]
learn_rate = [0.000075]
#epochs = [10]
#momentum = [0]
#dropout_rate = [0.35]
#activation = ['tanh', 'elu', 'sigmoid', 'linear']
#neurons = [16]
#leaky_relu_alpha = [0.2]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#scoring = {'precision_micro', 'precision_macro', 
#           'f1_micro', 'f1_macro', 
#           'recall_micro', 'recall_macro' 
#           'accuracy', 'balanced_accuracy'
#          }
#'accuracy', 'balanced_accuracy', 'precision', f1, 'recall'
param_grid = dict(batch_size=batch_size, 
                  learn_rate=learn_rate, 
#                   epochs=epochs,
#                   momentum=momentum, 
#                   dropout_rate=dropout_rate, 
#                   neurons=neurons, 
#                   leaky_relu_alpha=leaky_relu_alpha, 
#                   activation=activation
                  )
print(sorted(sklearn.metrics.SCORERS.keys()))
print('Training')
grid = GridSearchCV(estimator=model, param_grid=param_grid, #scoring=scoring, 
                    refit='accuracy', 
                    return_train_score='True', 
                    verbose=50,
                    n_jobs=2)

# Set early stopping criteria, ensure a reasonable number of epochs have
# passed if using early stopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              min_delta=0.001, 
                              patience=50, 
                              verbose=50, 
                              mode='auto', 
                              restore_best_weights=True)

print('Fit the DNN')
history = History()
grid_result = grid.fit(X_train, Y_train, 
                        callbacks=[history, earlyStopping],
                        validation_data=(X_test, Y_test),
                        shuffle=True)
print(history.history.keys())

print("Training Results")
nn_results = np.array(grid.predict(X_train))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
params = grid_result.cv_results_['params']

# Scores like this are vital to check your model isnt overfitting
print(sorted(sklearn.metrics.SCORERS.keys()))
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print('\n')
print(history.history.keys())
print('\n')
print("Loss is", history.history['loss'])
print("Val Loss is", history.history['val_loss'])
print('\n')
print("Accuracy is", history.history['categorical_accuracy'])
print('\n')
print("Val Accuracy is", history.history['val_categorical_accuracy'])

print('\n')
print('\n')

# Score the model - This first one is training accuracy, not to be used
# as a measure of the effectivness of the model, just as an indicator for
# things like overfitting
training_matrix = confusion_matrix(y_train, nn_results)
print(training_matrix)
ntraining_matrix = training_matrix / training_matrix.sum(axis=1)[:, np.newaxis]
cm = training_matrix
print(ntraining_matrix)
print("Categorical Accuracy:", accuracy_score(y_train, nn_results))
print(classification_report(y_train, nn_results))

tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
tn = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    tn.append(sum(sum(temp)))

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * ( (precision * recall) / (precision + recall) )

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Binary Rates: ")
print("FP is:", fp)
print("FN is:", fn)
print("TP is:", tp)
print("TN is:", tn)
print("Total is:", np.sum(cm))
print('\n')
print('\n')

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
params = grid_result.cv_results_['params']

#############################################################################
print('\n')
print('\n')
#############################################################################

# Real results here after predicting unseen data
print("Testing Results")
nn_results = np.array(grid.predict(X_eval))

training_matrix = confusion_matrix(y_eval, nn_results)
print(training_matrix)
ntraining_matrix = training_matrix / training_matrix.sum(axis=1)[:, np.newaxis]
cm = training_matrix
print(ntraining_matrix)

# Results created in two different ways to ensure they are correct
print("Categorical Accuracy:", accuracy_score(y_eval, nn_results))
print(classification_report(y_eval, nn_results))

tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
tn = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    tn.append(sum(sum(temp)))

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * ( (precision * recall) / (precision + recall) )

# The below should match the classification report
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Binary Rates: ")
print("FP is:", fp)
print("FN is:", fn)
print("TP is:", tp)
print("TN is:", tn)
print("Total is:", np.sum(cm))
print('\n')
print('\n')



print("*************************************")
print("*************************************")
print("************DECISION TREE************")
print("*************************************")
print("*************************************")
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train, y_train)

dt_results = dt.predict(X_test)
#dt_resultsnp = np.array(dt_results)

print("Results are:")
print(dt_results)

training_matrix = confusion_matrix(y_test, dt_results)
print(training_matrix)
ntraining_matrix = training_matrix / training_matrix.sum(axis=1)[:, np.newaxis]
cm = training_matrix
print(ntraining_matrix)
print("Categorical Accuracy:", accuracy_score(y_test, dt_results))
print(classification_report(y_test, dt_results))

tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
tn = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    tn.append(sum(sum(temp)))

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * ( (precision * recall) / (precision + recall) )

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Binary Rates: ")
print("FP is:", fp)
print("FN is:", fn)
print("TP is:", tp)
print("TN is:", tn)
print("Total is:", np.sum(cm))
print('\n')
print('\n')


print("*************************************")
print("*************************************")
print("************RANDOM FOREST************")
print("*************************************")
print("*************************************")


print("Random Forest")
rf = RandomForestClassifier(max_depth=10)
rf.fit(X_train, y_train)

rf_results = rf.predict(X_test)
rf_resultsnp = np.array(rf_results)

training_matrix = confusion_matrix(y_test, rf_results)
print(training_matrix)
ntraining_matrix = training_matrix / training_matrix.sum(axis=1)[:, np.newaxis]
cm = training_matrix
print(ntraining_matrix)
print("Categorical Accuracy:", accuracy_score(y_test, rf_results))
print(classification_report(y_test, rf_results))

tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
tn = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    tn.append(sum(sum(temp)))

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * ( (precision * recall) / (precision + recall) )

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Binary Rates: ")
print("FP is:", fp)
print("FN is:", fn)
print("TP is:", tp)
print("TN is:", tn)
print("Total is:", np.sum(cm))
print('\n')
print('\n')


print("*************************************")
print("*************************************")
print("************Linear SVM ************")
print("*************************************")
print("*************************************")

lin_clf = svm.LinearSVC()
lin_clf = lin_clf.fit(X_train, y_train)

lin_clfresults = lin_clf.predict(X_test)
lin_clfresultsnp = np.array(lin_clfresults)

training_matrix = confusion_matrix(y_test, lin_clfresults)
print(training_matrix)
ntraining_matrix = training_matrix / training_matrix.sum(axis=1)[:, np.newaxis]
cm = training_matrix
print(ntraining_matrix)
print("Categorical Accuracy:", accuracy_score(y_test, lin_clfresults))
print(classification_report(y_test, lin_clfresults))

tp = np.diag(cm)
fp = np.sum(cm, axis=0) - tp
fn = np.sum(cm, axis=1) - tp
tn = []
for i in range(num_classes):
    temp = np.delete(cm, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    tn.append(sum(sum(temp)))

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * ( (precision * recall) / (precision + recall) )

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Binary Rates: ")
print("FP is:", fp)
print("FN is:", fn)
print("TP is:", tp)
print("TN is:", tn)
print("Total is:", np.sum(cm))
print('\n')
print('\n')

