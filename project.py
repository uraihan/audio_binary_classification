import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,dct
import librosa
from scipy import signal
import soundfile as sf
from librosa.display import specshow
import glob

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from sklearn.metrics import accuracy_score, precision_score, recall_score


carTrain = glob.glob("cars/train/*.wav")
carTest = glob.glob("cars/test/*.wav")

tramTrain = glob.glob("trams/train/*.wav")
tramTest = glob.glob("trams/test/*.wav")

dataset=[]
labels = []

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

car_dataset, car_label  = importFiles(carTrain, 0)
car_test_dataset, car_test_label = importFiles(carTest, 0)
tram_dataset, tram_label = importFiles(tramTrain, 1)
tram_test_dataset, tram_test_label = importFiles(tramTest, 1)

dataset = np.concatenate([tram_dataset, car_dataset], axis=0)
labels = np.concatenate([tram_label, car_label], axis=0)

dataset_test = np.concatenate([tram_test_dataset,car_test_dataset], axis=0)
labels_test = np.concatenate([tram_test_label, car_test_label], axis=0)


fs = 44000
features=[]

fs = 44000
f2=[]

def extractFeatures(dataset,model):
    features=[]
    features2=[]
    for audio in dataset:
        mfccs = librosa.feature.mfcc(y=np.asarray(audio), sr=fs, n_mfcc=40)
        #features.append(mfccs)
    
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
        
        combined_features = np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
                                       np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                                       np.mean(spectral_centroid), np.std(spectral_centroid),
                                       np.mean(spectral_contrast), np.std(spectral_contrast),
                                       np.mean(zerocrossing_rate), np.std(zerocrossing_rate),
                                       np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        combined_features_for_CNN = np.hstack([spectral_bandwidth, spectral_centroid])
        combined_features_for_CNN2= np.hstack([zerocrossing_rate, spectral_rolloff])
        combined_features2=np.vstack([combined_features_for_CNN,combined_features_for_CNN2])

        if model == "KNN":
            features.append(combined_features)
        else:
            features.append(combined_features2)

    return features

features =extractFeatures(dataset,"KNN")
features2 =extractFeatures(dataset,"CNN")

features_test=extractFeatures(dataset_test,"KNN")
features_test2=extractFeatures(dataset_test,"CNN")

features= np.asarray(features)
features2= np.asarray(features2)

features_test= np.asarray(features_test)
features_test2= np.asarray(features_test2)

# # all
from sklearn.metrics import accuracy_score, precision_score, recall_score

# change is needed here: 
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)



input_shape = (2, 432, 1)

model = Sequential()
model.add(Conv2D(4, kernel_size=(2,2), activation='relu', input_shape=input_shape))
#model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(x=features2,y=labels,batch_size=5 ,epochs=10,validation_split=0.2,shuffle=True)

output=out1= model.predict(features_test2)

predictions = [1 if x > 0.5 else 0 for x in output]

accuracy2 = accuracy_score(labels_test, predictions)
precision2 = precision_score(labels_test, predictions)
recall2 = recall_score(labels_test, predictions)

print("Nearest Neighbour:")
print("Precision:", precision)
print("Recall:", recall)

print("CNN:")
print("Accuracy:", accuracy2)
print("Precision:", precision2)
print("Recall:", recall2)