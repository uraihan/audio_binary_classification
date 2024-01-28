# Introduction to Audio Processing - Project Work

This repository contains Python script and Jupyter Notebook for Project Work of Introduction to Audio Processing course.

## Goal
The goal was to: 1) Design a simple audio binary classifier solution to classify between trams and cars sounds, and 2) Compare the performance and building experience between different classifier solutions, in this case between Nearest-Neighbour and CNN classifier.

## Dataset
We used 200 audio files of both cars and trams in Tampere, Finland, collected by ourselves and from the Freesound platform. Audio collected by ourselves are treated as the test data, while audio collected from the Freesound platform are treated as training data.

## Classifiers
### Nearest-Neighbour
The Nearest-Neighbour classifier was implemented using a simple Scikit-learn's `KNeighborsClassifier`.

### CNN
The CNN classifier was originally implemented using Tensorflow.
