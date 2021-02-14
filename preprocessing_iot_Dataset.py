import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot
#Libraries for printing tables in readable format
from tabulate import tabulate

#Library for creating an excel sheet
import xlsxwriter
import operator
import seaborn as sns
import joblib

#Feature selection library
from featureselectionlibrary import featureSelectionUsingTheilU
from featureselectionlibrary import featureSelectionUsingChisquaredTest
from featureselectionlibrary import featureSelectionUsingRandomForestClassifier
from featureselectionlibrary import featureSelectionUsingExtraTreesClassifier

#feature encoding library
from featureencodinglibrary import featureEncodingUsingOneHotEncoder
from featureencodinglibrary import featureEncodingUsingLabelEncoder
from featureencodinglibrary import featureEncodingUsingBinaryEncoder
from featureencodinglibrary import featureEncodingUsingFrequencyEncoder

#feature scaling library
from featurescalinglibrary import featureScalingUsingMinMaxScaler
from featurescalinglibrary import featureScalingUsingStandardScalar
from featurescalinglibrary import featureScalingUsingBinarizer
from featurescalinglibrary import featureScalingUsingNormalizer
#feature scaling Library
from classificationlibrary import classifyUsingDecisionTreeClassifier
from classificationlibrary import classifyUsingLogisticRegression
from classificationlibrary import classifyUsingLinearDiscriminantAnalysis
from classificationlibrary import classifyUsingGaussianNB
from classificationlibrary import classifyUsingRandomForestClassifier
from classificationlibrary import classifyUsingExtraTreesClassifier
from classificationlibrary import classifyUsingKNNClassifier
from classificationlibrary import findingOptimumNumberOfNeighboursForKNN



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def shuffle_in_unison(a,b):

    assert len(a)==len(b)
    c = np.arange(len(a))
    np.random.shuffle(c)
    return a[c],b[c]


def getPathToTrainingAndTestingDataSets():
	trainingFileNameWithAbsolutePath = "/Users/jamil/Desktop/IDS-IoT-NSL-KDD/Dataset.csv"
	testingFileNameWithAbsolutePath = "/Users/jamil/Desktop/IDS-NSL-KDD/Autoencoder/KDDTest-21.csv"
	return trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath

def loadCSV (fileNameWithAbsolutePath):
    dataSet = pd.read_csv(fileNameWithAbsolutePath)
    return dataSet

def getLabelName():
	return 'attack_type'

def printList (list,heading):
    for i in range(0, len(list)):
        list[i] = str(list[i])
    if len(list)>0:
        print(tabulate([i.strip("[]").split(", ") for i in list], headers=[heading], tablefmt='orgtbl')+"\n")


#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    anyMissingValuesInTheDataset = dataSet.isnull().values.any()
    return anyMissingValuesInTheDataset

#This function is used to check for duplicate records in a given dataSet
#if duplicate are present it will
def checkForDulicateRecords (dataSet):
    totalRecordsInDataset = len(dataSet.index)
    numberOfUniqueRecordsInDataset = len(dataSet.drop_duplicates().index)
    anyDuplicateRecordsInTheDataset = False if totalRecordsInDataset == numberOfUniqueRecordsInDataset else True
    print('Total number of records in the dataset: {}\nUnique records in the dataset: {}'.format(totalRecordsInDataset,numberOfUniqueRecordsInDataset))
    return anyDuplicateRecordsInTheDataset

#Split the complete dataSet into training dataSet and testing dataSet
def splitCompleteDataSetIntoTrainingSetAndTestingSet(completeDataSet):
	labelName = getLabelName()
	label = completeDataSet[labelName]
	features = completeDataSet.drop(labelName,axis=1)
	featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet=train_test_split(features,label,test_size=0.1, random_state=42)
	print("features.shape: ",features.shape)
	print("label.shape: ",label.shape)
	return featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet




def getStatisticsOfData (dataSet):
    print("***** Start checking the statistics of the dataSet *****\n")

    labelName = getLabelName()
    #Number of rows and columns in the dataset
    print("***** Shape (number of rows and columns) in the dataset: ", dataSet.shape)

    #Total number of features in the dataset
    numberOfColumnsInTheDataset = len(dataSet.drop([labelName],axis=1).columns)
    #numberOfColumnsInTheDataset = len(dataSet.columns)
    print("***** Total number of features in the dataset: ",numberOfColumnsInTheDataset)

    #Total number of categorical featuers in the dataset
    categoricalFeaturesInTheDataset = list(set(dataSet.drop([labelName],axis=1).columns) - set(dataSet.drop([labelName],axis=1)._get_numeric_data().columns))
    #categoricalFeaturesInTheDataset = list(set(dataSet.columns) - set(dataSet._get_numeric_data().columns))
    print("***** Number of categorical features in the dataset: ",len(categoricalFeaturesInTheDataset))

    #Total number of numerical features in the dataset
    numericalFeaturesInTheDataset = list(dataSet.drop([labelName],axis=1)._get_numeric_data().columns)
    #numericalFeaturesInTheDataset = list(dataSet._get_numeric_data().columns)
    print("***** Number of numerical features in the dataset: ",len(numericalFeaturesInTheDataset))

    #Names of categorical features in the dataset
    print("\n***** Names of categorical features in dataset *****\n")
    printList(categoricalFeaturesInTheDataset,'Categorical features in dataset')

    #Names of numerical features in the dataset
    print("\n***** Names of numerical features in dataset *****\n")
    printList(numericalFeaturesInTheDataset,'Numerical features in the dataset')

    #Checking for any missing values in the data set
    anyMissingValuesInTheDataset = checkForMissingValues(dataSet)
    print("\n***** Are there any missing values in the data set: ", anyMissingValuesInTheDataset)

    anyDuplicateRecordsInTheDataset = checkForDulicateRecords(dataSet)
    print("\n***** Are there any duplicate records in the data set: ", anyDuplicateRecordsInTheDataset)
    #Check if there are any duplicate records in the data set
    #Not deleting of the duplicates
    if (anyDuplicateRecordsInTheDataset):
        dataSet = dataSet.drop_duplicates()
        print("Number of records in the dataSet after removing the duplicates: ", len(dataSet.index))

    #How many number of different values for label that are present in the dataset
    print('\n****** Number of different values for label that are present in the dataset: ',dataSet[labelName].nunique())
    #What are the different values for label in the dataset
    print('\n****** Here is the list of unique label types present in the dataset ***** \n')
    printList(list(dataSet[getLabelName()].unique()),'Unique label types in the dataset')

    #What are the different values in each of the categorical features in the dataset
    print('\n****** Here is the list of unique values present in each categorical feature in the dataset *****\n')
    categoricalFeaturesInTheDataset = list(set(dataSet.columns) - set(dataSet._get_numeric_data().columns))
    numericalFeaturesInTheDataset = list(dataSet._get_numeric_data().columns)
    for feature in categoricalFeaturesInTheDataset:
        uniq = np.unique(dataSet[feature])
        print('\n{}: {} '.format(feature,len(uniq)))
        printList(dataSet[feature].unique(),'distinct values')

    print('\n****** Label distribution in the dataset *****\n')
    print(dataSet[labelName].value_counts())
    print()

    print("\n***** End checking the statistics of the dataSet *****")


def defineArrayForPreProcessing():
	arrayOfModels = [
		[
			"ExtraTreesClassifier",
			"OneHotEncoder",
			"Standardization",
		]
	]
	# print(arrayOfModels)
	return arrayOfModels

def performPreprocessing(trainingDataSet, arrayOfModels):
    for i in range(0,len(arrayOfModels)):
        print('***************************************************************************************************************************')
        print('********************************************* Building Model-', i ,' As Below *************************************************')
        print('\t -- Feature Selection: \t ', arrayOfModels[i][0], ' \n\t -- Feature Encoding: \t ', arrayOfModels[i][1], ' \n\t -- Feature Scaling: \t ', arrayOfModels[i][2], '\n')

        trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath = getPathToTrainingAndTestingDataSets()
        #trainingDataSet = loadCSV(trainingFileNameWithAbsolutePath)
        #testingDataSet = loadCSV(testingFileNameWithAbsolutePath)

        labelName = getLabelName()
        label = trainingDataSet[labelName]

        #Combining the test and training datasets for preprocessing together, because we observed that in some datasets
        #the values in the categorical columns in test dataset and train dataset are being different this causes issues while
        #applying classification techniques
        completeDataSet = loadCSV(trainingFileNameWithAbsolutePath)

        #difficultyLevel = completeDataSet.pop('difficulty_level')

        print("completeDataSet.shape: ",completeDataSet.shape)
        print("completeDataSet.head: ",completeDataSet.head())

        #Feature Selection
        if arrayOfModels[i][0] == 'TheilsU':
            #Perform feature selection using TheilU
            completeDataSetAfterFeatuerSelection = featureSelectionUsingTheilU(completeDataSet)
        elif arrayOfModels[i][0] == 'Chi-SquaredTest':
            #Perform feature selection using Chi-squared Test
            completeDataSetAfterFeatuerSelection = featureSelectionUsingChisquaredTest(completeDataSet)
        elif arrayOfModels[i][0] == 'RandomForestClassifier':
            #Perform feature selection using RandomForestClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingRandomForestClassifier(completeDataSet)
        elif arrayOfModels[i][0] == 'ExtraTreesClassifier':
            #Perform feature selection using ExtraTreesClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingExtraTreesClassifier(completeDataSet)

        #Feature Encoding
        if arrayOfModels[i][1] == 'LabelEncoder':
            #Perform lable encoding to convert categorical values into label encoded features
            completeEncodedDataSet = featureEncodingUsingLabelEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'OneHotEncoder':
            #Perform OnHot encoding to convert categorical values into one-hot encoded features
            completeEncodedDataSet = featureEncodingUsingOneHotEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'FrequencyEncoder':
            #Perform Frequency encoding to convert categorical values into frequency encoded features
            completeEncodedDataSet = featureEncodingUsingFrequencyEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'BinaryEncoder':
            #Perform Binary encoding to convert categorical values into binary encoded features
            completeEncodedDataSet = featureEncodingUsingBinaryEncoder(completeDataSetAfterFeatuerSelection)

        #Feature Scaling
        if arrayOfModels[i][2] == 'Min-Max':
            #Perform MinMaxScaler to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingMinMaxScaler(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Binarizing':
            #Perform Binarizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingBinarizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Normalizing':
            #Perform Normalizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingNormalizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Standardization':
            #Perform Standardization to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingStandardScalar(completeEncodedDataSet)


    return 	completeEncodedAndScaledDataset


def autoEncoder(X_train,X_test):
    n_inputs = X_train.shape[1]

    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(n_inputs*2)(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(n_inputs)(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # bottleneck
    #n_bottleneck = n_inputs
    n_bottleneck = round(float(n_inputs) / 2.0) # The hidden layer has half number of input features
    bottleneck = Dense(n_bottleneck)(e)

    # define decoder, level 1
    d = Dense(n_inputs)(bottleneck)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(n_inputs*2)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # output layer
    output = Dense(n_inputs, activation='linear')(d)
    # define autoencoder model
    model = Model(inputs=visible, outputs=output)
    # compile autoencoder model
    model.compile(optimizer='adam', loss='mse')
    # plot the autoencoder
    #plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)
    # fit the autoencoder model to reconstruct input
    history = model.fit(X_train, X_train, epochs=20, batch_size=16, verbose=2, validation_data=(X_test,X_test))
    #plot loss

    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

    # define an encoder model (without the decoder)
    encoder = Model(inputs=visible, outputs=bottleneck)
    #plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
    # save the encoder to file
    encoder.save('encoder.h5')


def nn_model(trainx, trainy, valx,valy,bt_size,epochs, layers):
  model = Sequential()
  model.add(Dense(layers[0],activation='relu', input_shape=(trainx.shape[1],)))
  for l in layers[1:]:
    model.add(Dense(l, activation='relu' ))
    model.add(Dropout(0.30))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m,recall_m])
  #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  hist=model.fit(trainx, trainy, batch_size=bt_size, epochs=epochs, shuffle=True, validation_data=(valx,valy), verbose=True)
  model.save('dnn.h5')
  loss, accuracy, f1_score, precision, recall = model.evaluate(valx, valy, verbose=0)
  print("loss", loss,  "accuracy", accuracy*100, "f1_score", f1_score, "precision", precision, "recall", recall)
  return hist

trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath = getPathToTrainingAndTestingDataSets()

trainingDataSet = loadCSV(trainingFileNameWithAbsolutePath)
#difficultyLevel = trainingDataSet.pop('difficulty_level')
labelName = getLabelName()
label = trainingDataSet[labelName]
getStatisticsOfData(trainingDataSet)

# #Define file names and call loadCSV to load the CSV files
# testingDataSet = loadCSV(testingFileNameWithAbsolutePath)
# #difficultyLevel = testingDataSet.pop('difficulty_level')
# getStatisticsOfData(testingDataSet)
arrayOfModels = defineArrayForPreProcessing()
completeEncodedAndScaledDataset = performPreprocessing(trainingDataSet, arrayOfModels)
#
# #Split the complete dataSet into training dataSet and testing dataSet
featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet = splitCompleteDataSetIntoTrainingSetAndTestingSet(completeEncodedAndScaledDataset)
#
trainingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTrainingDataSet, labelInPreProcessedTrainingDataSet], axis=1, sort=False)
testingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTestingDataSet, labelInPreProcessedTestingDataSet], axis=1, sort=False)
#
#
X_train = trainingEncodedAndScaledDataset.drop('attack_type',axis=1)
y_train = trainingEncodedAndScaledDataset['attack_type']
X_test = testingEncodedAndScaledDataset.drop('attack_type',axis=1)
y_test = testingEncodedAndScaledDataset['attack_type']
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
#
X_trainNormal,X_trainMalicious=X_train[y_train=='Normal'],X_train[y_train!='Normal']
X_testNormal,X_testMalicious=X_test[y_test=='Normal'],X_test[y_test!='Normal']
# print(X_trainNormal.shape,X_trainMalicious.shape,X_testNormal.shape,X_testMalicious.shape)
#
#
X_trainNormal=X_trainNormal.to_numpy()
X_trainMalicious=X_trainMalicious.to_numpy()
X_testNormal=X_testNormal.to_numpy()
X_testMalicious=X_testMalicious.to_numpy()
print(X_trainNormal.shape,X_trainMalicious.shape,X_testNormal.shape,X_testMalicious.shape)
#
X_train_final=np.append(X_trainNormal, X_trainMalicious, axis=0)
y_trainNormal=np.zeros(X_trainNormal.shape[0])
y_trainMalicious=np.ones(X_trainMalicious.shape[0])
y_train_final=np.append(y_trainNormal,y_trainMalicious)
print(X_train_final.shape, y_train_final.shape)
#
X_test_final=np.append(X_testNormal, X_testMalicious, axis=0)
y_testNormal=np.zeros(X_testNormal.shape[0])
y_testMalicious=np.ones(X_testMalicious.shape[0])
y_test_final=np.append(y_testNormal,y_testMalicious)
print(X_test_final.shape, y_test_final.shape)
#
X_train_final,y_train_final=shuffle_in_unison(X_train_final,y_train_final)
X_test_final,y_test_final=shuffle_in_unison(X_test_final,y_test_final)

# # print(X_train_final[:10])
# # print(y_train_final[:10])
# # print(X_test_final[:10])
# # print(y_test_final[:10])
#
#
autoEncoder(X_train_final,X_test_final)
#
# # load the model from file
encoder = load_model('encoder.h5')
# # encode the train data
X_train_final_encode = encoder.predict(X_train_final)
# # encode the test data
X_test_final_encode = encoder.predict(X_test_final)
#
layers=[1000,500,300,100,50,10]
hist = nn_model(X_train_final_encode, y_train_final, X_test_final_encode, y_test_final,16,50,layers)
print('MAX Accuracy during training: ',max(hist.history['accuracy'])*100)
print('MAX Accuracy during validation: ',max(hist.history['val_accuracy'])*100)
plt.plot(range(50), hist.history['accuracy'], 'r', label='Train acc')
plt.plot(range(50), hist.history['val_accuracy'], 'b', label='Test acc')
plt.legend()
plt.show()
#
# #Following code is for encoder
