import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,dct
import librosa
from scipy import signal
import soundfile as sf
from librosa.display import specshow
import glob

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


def importFiles(files, label):
    dataset = []
    labels = []
    for file in files:
        data,sr = librosa.load(file)
        data = librosa.effects.trim(data, top_db=20, frame_length=1024, hop_length=512)[0] 
        # print(data.size)
        # print(data.shape)# Desired length in samples
        desired_length = sr * 5
        # # Initialize a new array of zeros with the desired length
        fixed_length_data = np.zeros(desired_length)
        #  Check the length of the original data
        original_length = len(data)
        # # If original data is longer than desired length, truncate it
        # # If it is shorter, pad with zeros
        if original_length > desired_length:
            fixed_length_data = data[:desired_length]
        else:
            fixed_length_data[:original_length] = data
        # # Now use fixed_length_data as your adjusted data
        data = fixed_length_data

        labels.append(label)
        dataset.append(data)

    return dataset,labels

def extractFeatures(dataset, model):
    features = []
    fs = 44000

    for audio in dataset:
        mfccs = librosa.feature.mfcc(y=np.asarray(audio), sr=fs, n_mfcc=40)
        # spectral spread
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=np.asarray(audio), sr=fs)

        # spectral energy
        spectral_centroid = librosa.feature.spectral_centroid(y=np.asarray(audio), sr=fs)

        # spectral density
        spectral_contrast = librosa.feature.spectral_contrast(y=np.asarray(audio), sr=fs)

        #  rate of sign-changes in the signal
        zerocrossing_rate = librosa.feature.zero_crossing_rate(y=np.asarray(audio))

        #  frequency below which a certain percentage of the power spectrum is concentrated
        spectral_rolloff = librosa.feature.spectral_rolloff(y=np.asarray(audio), sr=fs)
        
        if model == "KNN":
            combined_features = np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
                                       np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                                       np.mean(spectral_centroid), np.std(spectral_centroid),
                                       np.mean(spectral_contrast), np.std(spectral_contrast),
                                       np.mean(zerocrossing_rate), np.std(zerocrossing_rate),
                                       np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            
            # mfccs = np.reshape(mfccs, (1,-1))
            # spectral_contrast = np.reshape(spectral_contrast, (1,-1))
            # combined_features = np.hstack([spectral_bandwidth, spectral_centroid, zerocrossing_rate, spectral_rolloff, mfccs, spectral_contrast])
            features.append(combined_features)
        
        elif model == "CNN":
            combined_features_for_CNN = np.hstack([spectral_bandwidth, spectral_centroid])
            combined_features_for_CNN2= np.hstack([zerocrossing_rate, spectral_rolloff])
            combined_features = np.vstack([combined_features_for_CNN,combined_features_for_CNN2])
            
            features.append(combined_features)
        else:
            raise Exception("model is not recognizable.")

    features = np.array(features)
    return features


carTrain = glob.glob("cars/train/*")
carTest = glob.glob("cars/test/*")

tramTrain = glob.glob("trams/train/*")
tramTest = glob.glob("trams/test/*")

# import audio from files
car_dataset, car_label  = importFiles(carTrain, 0)
car_test_dataset, car_test_label = importFiles(carTest, 0)
tram_dataset, tram_label = importFiles(tramTrain, 1)
tram_test_dataset, tram_test_label = importFiles(tramTest, 1)

# combine tram and cars dataset
dataset_train = np.concatenate([tram_dataset, car_dataset], axis=0)
labels_train = np.concatenate([tram_label, car_label], axis=0)

dataset_test = np.concatenate([tram_test_dataset,car_test_dataset], axis=0)
labels_test = np.concatenate([tram_test_label, car_test_label], axis=0)

# extract features from the dataset
features_train = extractFeatures(dataset_train, "KNN")
features_test = extractFeatures(dataset_test, "KNN")

features_train = features_train.reshape((features_train.shape[0], -1))
features_test = features_test.reshape((features_test.shape[0], -1))

## K-Nearest Neighbour Classifier, with all features

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train, labels_train)

y_pred = knn.predict(features_test)

precision_knn = precision_score(labels_test, y_pred)
recall_knn = recall_score(labels_test, y_pred)
accuracy_knn = accuracy_score(labels_test, y_pred)

## Convolutional Neural Network

# extract features from the dataset
features_train2 = extractFeatures(dataset_train, "CNN")
features_test2 = extractFeatures(dataset_test, "CNN")

input_shape = (2, 432, 1)

model = Sequential()
model.add(Conv2D(4, kernel_size=(2,2), activation='relu', input_shape=input_shape))
#model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x=features_train2,y=labels_train,batch_size=5,epochs=10,validation_split=0.2,shuffle=True)

output = model.predict(features_test2)
predictions = [1 if x > 0.5 else 0 for x in output]

accuracy_cnn = accuracy_score(labels_test, predictions)
precision_cnn = precision_score(labels_test, predictions)
recall_cnn = recall_score(labels_test, predictions)


print("Nearest Neighbour:")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)

print("CNN:")
print("Accuracy:", accuracy_cnn)
print("Precision:", precision_cnn)
print("Recall:", recall_cnn)

print("Testing with audio number 10 from test data (0 means car, 1 means tram)")
print("Real label: ", labels_test[9])

print("KNN Prediction")
print("Prediction: ", y_pred[9])

print("CNN Prediction")
print("Prediction: ", output[9])