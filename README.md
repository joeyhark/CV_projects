# Emotion Classification

The goal of this project is to explore the use of deep neural networks for emotion classification on facial images. Landmark convolutonal architectures like [VGG16](https://arxiv.org/abs/1409.1556) are implemented in their original and modified structures. Variations of [GoogLeNet](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) are also used. All networks are implemented using Keras, TensorFlow, and Python and run on [Northeastern's high performance computing computing resource](https://rc-docs.northeastern.edu/en/latest/welcome/welcome.html) to allow TensorFlow to use a GPU.

This project began in summer 2020, when the COVID pandemic began greatly impacting all facets of life. Mask wearing and switching from in-person to video communication meant body language and sometimes facial expressions, were obscured. This made percieving emotion more difficult, especially for those who already struggle with this task. The project is motivated by this problem, but its benefits would extend to security, psychology, and other fields. The specific aim is to find or modify a neural network that can correctly classify anger, disgust, fear, happiness, neutrality, sadness, and suprise given a facial image. Tuning hyperparameters is explored through learning rate finders and early stopping. Results are currently displayed in graphical and tabular forms, as shown below:

Graph                  |  Report
:-------------------------:|:-------------------------:
<img src="https://github.com/joeyhark/emotion_classification/blob/master/results/smaller_VGGNet2_FERC.png" width="400">  |  <img src="https://github.com/joeyhark/emotion_classification/blob/master/results/smaller_VGGNet2_FERC_report.png" width="400">

Data currently in use is obtained from [Kaggle's Facial Expression Recognition Challenge (FERC)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and [Karolinska Institute's Directed Emotional Faces (KDEF) set](https://www.kdef.se/home/aboutKDEF.html). Preprocessing is adapted by network for optimal results. The project began by using the [MNIST](http://yann.lecun.com/exdb/mnist/) and [fashion MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master/data) datasets to benchmark the networks. The project is currently in an experimental phase where networks are being adapted to increase accuracy.

**Contents:**  
*Functional, In Use*  
LRF_and_architectures  
&ensp;&ensp;`clr_callback.py` - [Brad Kenstler's cyclical learning rate implementation](https://github.com/bckenstler/CLR)   
&ensp;&ensp;`config_FERC.py` - hyperparameters, input/output paths, and settings for training using FERC  
&ensp;&ensp;`learning_rate_finder.py` - [Adrian Rosebrock's implementation](https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/) of [Arun Mayia's learning rate finder](https://github.com/amaiya/ktrain/blob/master/ktrain/lroptimize/lrfinder.py)  
&ensp;&ensp;`minigooglenet.py` - lightweight implementation of GoogLeNet  
&ensp;&ensp;`smaller_VGGNet.py` - alterned convolutional architecture based on VGG16  
&ensp;&ensp;`smaller_VGGNet2.py` - alterned convolutional architecture based on VGG16  
&ensp;&ensp;`smaller_VGGNet3.py` - alterned convolutional architecture based on VGG16  
data  
&ensp;&ensp;FERC_sorted - images from FERC sorted by emotion class  
&ensp;&ensp;KDEF_orig - images from KDEF sorted by subject  
&ensp;&ensp;KDEF_sorted - images from KDEF sorted by emotion class  
`predict_smaller_VGGNet.py` - makes single image or multi image emotion class predictions from trained VGG models  
`sort_KDEF_orig.py` - sorts original KDEF dataset by labelled emotion class  
`train_GoogLeNet_FERC.py` - trains and tests GoogLeNet implementation on FERC with optional learning rate finder, yields graphical and tabular results  
`train_VGG16_SGD.py` - trains head of VGG16 using pre-trained ImageNet weights in network body, yields accuracy/loss plot  
`train_smaller_VGGNet.py` - trains altered VGG16 implementation, yields accuracy/loss plot  
`train_smaller_VGGNet.py` - trains altered VGG16 implementation with optional learning rate finder, yields accuracy/loss plot  
results - output directory for accuracy/loss plots and graphical and tabular results  

*Functional, Not In Use*  
start_stop_training  
&ensp;&ensp;callbacks  
&ensp;&ensp;&ensp;&ensp;`epochcheckpoint.py` - TensorFlow callback for saving model during training  
&ensp;&ensp;&ensp;&ensp;`epochcheckpoint.py` - TensorFlow callback for saving training accuracy and loss graphically during training  

*Functional, Obsolete*   
LRF_and_architectures  
&ensp;&ensp;`LeNet_MNIST.py` -  [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) implementation for MNIST 
&ensp;&ensp;`config_MNIST.py` -  hyperparameters, input/output paths, and settings for training using MNIST  
start_stop_training  
&ensp;&ensp;nn  
&ensp;&ensp;&ensp;&ensp;`resnet.py` - residual neural network implementation  
`predict_LeNet_MNIST.py` - tests LeNet on MNIST dataset, yields tabular results and example prediction  
`train_GoogLeNet_fashion_MNIST.py` - trains GoogLeNet implementation on fashion MNIST with optional learning rate finder, yields accuracy/loss plot  
`train_LeNet_MNIST.py` - trains LeNet on MNIST, yields final accuracy and loss

*Non-Functional*  
`predict_VGG16_SGD(unfinished)` - aims to predict emotion classes for VGG16 and yield tabular results  

**Issues/Improvements:**  
- [ ] Seperate MNIST implementations from emotion classification more explicitly.
- [ ] Adapt scripts to `train_GoogLeNet_FERC.py` format to train and predict in one run and include learning rate finder option.
- [ ] Organize all callbacks.
- [ ] Implement early stopping functionality without using start stop callback.
- [ ] Implement command line arguments in all scripts to take hyperparameters, dataset, LRF option, start stop option, and early stop option as input.
- [ ] Modularize pre-processing steps and call pre-processing based on dataset user input.
