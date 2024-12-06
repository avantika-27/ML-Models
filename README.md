Objective of the project
This project focuses on implementing and comparing various machine learning models to perform binary classification tasks on two well-known datasets: MNIST and CIFAR-10.

Implemented Models

Models for MNIST Dataset
1. Logistic Regression
2. Kernel Support Vector Machine (Kernel SVM)

Models for CIFAR-10 Dataset
3. PCA + Fully Connected Neural Network (FNN)
4. 3-Layer Convolutional Neural Network (CNN)

Environment
* The Logistic Regression, Kernel SVM, and PCA+FNN models are implemented using libraries compatible with Python 11 can run on local machine or on Google collab
* The 3-Layer CNN model uses TensorFlow. Therefore, it should be run in Google Collab.

Libraries
Below are the libraries required for the respective models:
1. scikit-learn (sklearn):
   * For Logistic Regression, Kernel SVM, PCA, and basic preprocessing.
2. numpy:
   * For numerical operations and array manipulations.
3. matplotlib:
   * For visualization of results, including graphs and confusion matrices.
Additional Libraries
4. tensorflow:
   * For building and training the CNN model.

Install all required libraries using the following command:
pip install numpy scikit-learn matplotlib tensorflow

Execution Instructions

1.Running in Google Collab
* Logistic Regression (MNIST)
* Kernel SVM (MNIST)
* PCA + FNN (CIFAR-10)
* 3-Layer CNN model should be executed in Google Collab due to its dependency on TensorFlow.

Steps:
1. Open the files in Google Collab.
2. Ensure that the MINST/CIFAR-10 dataset is downloaded or automatically loaded using TensorFlow’s dataset utilities or sklearn libraries.
3. Execute the cells sequentially to train and evaluate the  model.
