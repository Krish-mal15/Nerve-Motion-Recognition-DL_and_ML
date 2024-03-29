# Nerve-Motion-Recognition-DL_and_ML


## Overview
The goal of this project is to be able to take brain data and convert it into movements. I made two deep learning models using the Feedforward Neural Networks which yielded mediocre results at about a 70% accuracy each, achieved in Keras and PyTorch. I also implemented a Sci-kit learn model using the Random Forest Classifier model. This yieled great results, resulting in a 98.7% accuracy! In the future, if I could get my hands on a device which could constantly read data from the brain, these models could turn them into movements on a robotic limb.


## Data:
- Kaggle: https://www.kaggle.com/datasets/sojanprajapati/emg-signal-for-gesture-recognition/data
- CSV File
- Dropped the time and label columns
- Converted to Tensors in deep learning models
- The output from 8 sensors on a neural-sensing bracelet placed on the forearm of the subject was sectioned into the following classes (gestures performed):
  1 - hand at rest,
  2 - hand clenched in a fist,
  3 - wrist flexion,
  4 – wrist extension,
  5 – radial deviations,
  6 - ulnar deviations,
  7 - extended palm (the gesture was not performed by all subjects)
  

## Deep Learning Models (Keras Model and PyTorch Model):
Both of these models follow the same architecture of a 2 layer Feed-Forward Neural Network. Consequently, they bothresukted in a 70% accuracy. In short, these models apply a linear layer followed by a ReLU activation twice. Following this, a softmax function is applied and the modle is trained on 8 epochs using the cross-entropy loss and adam optimizer. The only difference is the Pytorch model uses a stochastic gradient descent optimizer which showed the same results as using an adam optimizer. The below image is a good representation of a feed forward neural network. Credit: https://www.linkedin.com/pulse/feed-forward-neural-network-prashant-piyush
I faced some issues regarding my input which I had to map to the 8 classes outputted in the dataset and configure my hidden layers accordingly as well.



![1538057889628](https://github.com/Krish-mal15/Nerve-Motion-Recognition-DL_and_ML/assets/156712020/0bf7b182-c9f9-4cb9-ad84-0cf3f44d9a01)



In very short terms, a feedforward neural network as such mine, takes an input number of neurons, while continuosly moving forward and maps it to different output dimensions (linearizing) and then finally outputs the number of classes to make a prediction on the most likely class.  For more detailed info, see: https://www.turing.com/kb/mathematical-formulation-of-feed-forward-neural-network


## Machine Learning Model (Scikit-learn)
As I understand that machine learning may be a more suitable option for classification in data of numbers, I implemented a basic machine learning model using scikit-learn using the Random Forest Classifier. This model has proven best for machine learning classification tasks despite its high computational cost. Essentially, data is provided to a specified amount of decision trees which are assigned a random feature from the data. The data is processed through multiple complex algorithms. This process is repeated multiple times. Following this, for the classification task, a "majority vote" occurs in which the algorithm will conclude a single class based on results from the separate trees. In my application, the electrode data is fed into 100 trees which is then mapped to the outputs and then based on continuous inputted test results, the decision trees will "agree" on a certain class which is the predicted motion of the arm and output that. Image Source and further information: https://www.ibm.com/topics/random-forest


![image](https://github.com/Krish-mal15/Nerve-Motion-Recognition-DL_and_ML/assets/156712020/4c8fb01b-8b5f-4f1b-b311-70f26d40dbe7)


