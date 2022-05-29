
# Real-Time-Emotion-Detection

This is an application developed for Microsoft Intern Engage 2022 Programme.
We all know facial emotions play a vital role in our day-to-day life.
 So, we need a system which is capable of recognizing our facial emotions and able to act accordingly. 

 


##  Model Building using Convolutional Neural Networks(CNN)

The convolutional neural network, or CNN for short, is a specialized type of neural network model designed for working with two-dimensional image data, although they can be used with one-dimensional and three-dimensional data.
The Convolutional Neural Network was built using TensorFlow, Keras, and OpenCV Python.
## Tech Stack Used
Kaggle Dataset-https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset



**To train the model**
```
python main.py  [-t] [--data [data_path]] [--epochs] [--learning_rate] [--batch_size]

--data                  Data folder that contains training and validation files
--train                 True when training
--epochs                Number of epochs
--learning_rate         Learning rate value
--batch_size            Training/validation batch size

-> This file is used for training the Deep Learning Neural Network.
The images are loaded into the file using Keras flow_from_directory() function. 
Each Image is preprocessed and converted to a NumPy array. 
ImageDataGenerator() function is used for Data Augmentation.

```

**To validate & Test the model**
```
python visualize.py [-t] [-c] [--data [data_path]] [--model [model_path]]

--data                  Data folder that contains test images and test CSV file
--model                 Path to pretrained model
--test_cc               Calculate the test accuracy
--cam                   Test the model in real-time with webcam connect via USB

-> This file uses OpenCV and Numpy. 
The Trained Model is loaded using Keras load_model() function. 
Using cv2.VideoCapture() function the video from webcam is captured.
haarcascade_frontalface_default.xml is used to detect the face of a person. 

  
```

 
* Jupyter Notebook to execute the code using webcam.       
* To view the application refer the code given above or run the code using jupyter notebook.

## Installation


* Anaconda3
* Libraries : pip install Keras , pip install tensorflow & pip install opencv-python


              
## Tutorial

https://www.youtube.com/watch?v=G1Uhs6NVi-M&t=24s
## Future Aspects

This emotion recognition software is now being developed by major tech firms aiming to incorporate such software to get a deep insight into unfiltered, real, and raw emotional responses to predefined digital content only through webcams. This way, companies will be able to accurately determine how their consumers respond and react when they use their website or their application. By gathering the data, the firm will be able to improve the user experience.
