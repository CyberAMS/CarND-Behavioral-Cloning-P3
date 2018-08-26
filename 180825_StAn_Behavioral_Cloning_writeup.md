# Project: Behavioral Cloning

This project has been prepared by Andre Strobel.

The goal of this project is to use a convolutional neural network (CNN) to steer a car around a simulated track. The CNN must learn its behavior from a user driving the car around the same track in the simulator.

Everything has been programmed in Python 3 using Tensorflow, Keras and the Udacity Self-Driving Car Engineer Simulator.

---

## Content

1. Setting up the simulator and training environment
    1. Data sources
    1. Hardware considerations
1. Capturing the behavioral training data
    1. Basic strategy
    1. Pre-processing of recovery data
    1. Augmentation of behavioral training data
1. Defining the model training pipeline using generators
    1. Generator creation
    1. Flexible model definition
1. Selecting the model architecture and hyperparameters
    1. Basic structure of the model
    1. Considered model variations
    1. Hyperparameter tuning
1. Evaluating the model bahavior
    1. What does underfitting and overfitting mean in this example?
    1. Which model architecture and what parameters worked best?
1. Discussion

[//]: # (Image References)

[image1]: docu_images/01_02_center_2018_08_18_06_11_22_467.jpg
[image2]: docu_images/01_02_center_2018_08_18_06_11_22_467_cropped.jpg
[image3]: docu_images/01_02_center_2018_08_18_06_11_22_467_flipped.jpg

---

## 1. Setting up the simulator and training environment

### 1. Data sources

My goal is to be able to work on similar projects using my own hardware in the future. Therefore, I did not use the provided workspace nor an [AWS instance](https://aws.amazon.com/).

I installed [Anaconda](https://anaconda.org/) and defined a `tensorflow_GPU` environment with a *Jupyter* *notebook* and *jupyterlab*. Then I installed [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) in this `tensorflow_GPU` environment.

```
conda create -n tensorflow_GPU pip python=3.5 anaconda
activate tensorflow_GPU
pip install --ignore-installed --upgrade tensorflow-gpu
pip install keras-gpu
```

In order to use the Keras `keras.utils.plot_model` function I also had to manually install the `pydot` and `graphviz` packages in the `tensorflow_GPU` environment. For `graphviz` to work on my Windows 10 computer, I also had to install the [GraphViz](https://www.graphviz.org/download/) tools and set the Windows path to point to these executables.

A Windows executable for the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) can be found [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip). In the following sections I will refer to this simply as *simulator*.

In order to run the *simulator* in conjunction with a provided `drive.py` Python script in *Autonomous Mode*, I also needed to manually install the `socketio` package in the `tensorflow_GPU` environment.

### 2. Hardware considerations

I strongly recommend using a GPU for running the *simulator* in conjunction with the `drive.py` Python script in *Autonomous Mode*. I am using an *NVIDIA GeForce GTX 1060 6 GB* graphics card for this. If I run the *simulator* without a GPU on an *Intel i7* platform with the fastest low resolution settings, I will experience response time issues between the *simulator* and the Python `drive.py` script. The Python `drive.py` script itself does not require a GPU.

In order to read images quickly from the hard disk during training, I highly recommend to use a fast *M.2 Solid-State Drive*. Compared to running it on a fast network attached storage this sped up the training process from many hours to a few minutes.

## 2. Capturing the behavioral training data

### 1. Basic strategy

Recording data in the *simulator* will save a sequence of images (center view, left view, right view) as well as the corresponding  measurements for steering angle, throttle, braking and speed.

The goal of this project is to have a car drive in the center of the provided track. Hence, collecting data of good center driving is essential. This will train the model on which steering angle to choose when seeing a specific image in the center view. The left and right views can be used to augment the training data with views that require a slight steering adjustment to get back to the center line. In the following the slight steering adjustment is defined by the parameter `steeroffset` (value greater or equal to 0). The following images show a left, center and right view of the same training data set. The steering angle `steering_angle` is a value between -1 and 1. A slightly negative value demands a slight steering to the left. The car uses maximum throttle for `throttle` (value between 0 and 1), no braking for `braking` (value between 0 and 1) and drives with a maximum speed `speed` (value between 0 and 30-ish).

<img src="docu_images/01_01_left_2018_08_18_06_11_22_467.jpg" width="30%"> <img src="docu_images/01_02_center_2018_08_18_06_11_22_467.jpg" width="30%"> <img src="docu_images/01_03_right_2018_08_18_06_11_22_467.jpg" width="30%">

```
left view: steering_angle = -0.03759398 + steeroffset
center view: steering_angle = -0.03759398
right view: steering_angle = -0.03759398 - steeroffset
throttle = 1
braking = 0
speed = 30.19014
```

The algorithm should focus on the road itself and not on objects in the environment. The basic strategy does not include to initiate specific driving actions when e.g. a specific tree is seen on the side of the road. Therefore, all the images will be cropped at the top and bottom as shown in the following.

![alt text][image1] ![alt text][image2]

### 2. Pre-processing of recovery data

Although the center line training includes center line driving and a soft version for recovery, the car will eventually drift away from the center line due to either varying initial conditions, changing environments or prediction errors. Therefore, a stronger version for recover is needed when the car gets close to the boundaries of the driveable track.

We need to train the model to steer away from the track boundary if the car gets too close to it. We can record the necessary steering angle based on the behavior of a real driver by driving on the track and continuously weaving from left to right. Then we pre-process this recovery data to only consider images that steer the car away from the boundary. We don't want to train the model to steer towards the track boundary. We also don't want to train the model to cross over the center line with a steering angle that moves the car away from the center line like we do during the weaving events.

As the measurements are recorded in a *\*.csv* format I conveniently used *Microsoft Excel* to mark the rows that contain valid recovery situations during the weaving events. A valid recovery situation is determined as being part of the first third of an event when the steering wheel clearly changes from left to right steering or the other way round. The second third would be considered crossing the center line and the last third would be considered steering towards the closest boundary. Here are examples for first, second and last third images of a single event as decribed before along with their steering angles.

<img src="docu_images/02_01_center_2018_08_18_06_36_35_877.jpg" width="30%"> <img src="docu_images/02_02_center_2018_08_18_06_36_37_839.jpg" width="30%"> <img src="docu_images/02_03_center_2018_08_18_06_36_39_145.jpg" width="30%">

```
first third: steering_angle = 0.2406015
second third: steering_angle = 0.2330827
last third: steering_angle = 0.2030075
```

The following table shows the columns of the *\*.csv* file and the values that they contain.

| Row number | Column D | Column E | Column F | Column G |
|------------|----------|----------|----------|----------|
| 6 | \<steering angle\> | \<throttle\> | \<braking\> | \<speed\> |

The next table shows the formulas I used to mark the valid recovery situations starting in row 6. Row 1 to 5 cannot contain valid situations. Column I is used to identify whether the new steering angle is part of a recovery (1 if yes and 0 if not) from the right (steering angle larger than the average of the last 5 events or steering angle negative) or the left (steering angle smaller than the average of the last 5 events or steering angle positive) side. Column J increases the counter for how many steps the most recent recovery already took by 1 if a recovery takes place - counting from the end to the beginning. Column K divides column J by 3 and is used by column L to count down by 1 step each row. Column M uses this information to return 1 if the current row is part of the first third of the latest recovery event and 0 otherwise.

| Row number | Column I | Column J | Column K | Column L | Column M |
|------------|----------|----------|----------|----------|----------|
| 6 | =IF(OR(AND(D6>=AVERAGE(D1:D5),D6<0),AND(D6<=AVERAGE(D1:D5),D6>0)),1,0) | =IF(I6=0,0,J7+I6) | =IF(J6=0,0,J6/3) | =IF(K6=0,0,IF(AND(K5=0,K6>0),K6,L5-1)) | =IF(L6>=1,1,0) |

Columns H to L are deleted before exporting the *Microsoft Excel* file to a *\*.csv'* file. Column M stays and is read as variable `btake` for each center, left and right view image.

### 3. Augmentation of behavioral training data

To further augment the training data I also drove the track in the opposite direction - using center line driving and weaving.

Each image can be used as is and horizontally flipped. The sign of the steering angle of the flipped image must also be flipped to maintain a valid training data set. Here is an example of a original and flipped image and their steering angles.

![alt text][image1] ![alt text][image3]

```
original center view: steering_angle = -0.03759398
flipped center view: steering_angle = 0.03759398
```

With the before mentioned training data the car tends to either stay in the center, drift to the boundaries or recover hard from the boundaries. This can be problematic when the car drives faster. Therefore, an additional medium recovery dataset was recorded in both directions by weaving around the center line. The same pre-processing was applied as decribed in the section before.

A further step would be to shift the image left and right, adjust the steering angle accordingly to encourage center driving  and regenerate the sides of the image using an auto-encoder that has been trained on the complete training dataset. This has not been implemented yet.

The total number of recorded training datasets is shown in the following table. The columns contain the numbers of center, left and right view images in original and flipped state. The rows contain the numbers for the individual recordings of track `track1`. Each loop on `track1` needed to be split into 2 separate recordings, because recording the full loop at once sometimes led to an error. The center line driving is recorded as `center` and `counter` for the opposite direction. The strong recovery weaving is recorded as `weave` and `ceave` for the opposite direction. The medium recovery weaving is recorded as `meave` and `deave` for the opposite direction. Although many invalid images are ignored during the recovery weaving recordings, the total number is still high, because these loops are driven at much slower speeds. 

```
Number of image files: 88020

                 center  centerflipped  left  leftflipped  right  rightflipped
track1_center1     1325           1325  1325         1325   1325          1325
track1_center2     1289           1289  1289         1289   1289          1289
track1_counter1    1287           1287  1287         1287   1287          1287
track1_counter2    1327           1327  1327         1327   1327          1327
track1_weave1      2106           2106  2106         2106   2106          2106
track1_weave2      1162           1162  1162         1162   1162          1162
track1_ceave1       631            631   631          631    631           631
track1_ceave2      1918           1918  1918         1918   1918          1918
track1_meave1      1064           1064  1064         1064   1064          1064
track1_meave2       857            857   857          857    857           857
track1_deave1       738            738   738          738    738           738
track1_deave2       966            966   966          966    966           966
```

## 3. Defining the model training pipeline using generators

I developed a very flexible model setup and training pipeline that uses generators for effective data handling. With this I can try different model configurations by simply varying parameters. The complete pipeline is based on the following Python packages, functions and objects. The example code sections that are listed below don't require all of them.

```python
# import packages
import glob
import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Flatten, Dropout, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from keras.utils import plot_model
import pandas as pd
```

### 1. Generator creation

The training datasets are determined by the function `get_data`. This function returns 3 lists: `imagefiles` contains all training image file names, `measurements` contains all measurements for the training images and `bmustflip` contains the info whether or not the image must be flipped. It also returns the size of the training images as `ysize` for the height in pixels and `xsize` for the width in pixels.

The actual content of the image files is loaded and pre-processed with the `get_data_generator` function. It loops infinitely over the dataset and shuffles it at the beginning before going through it again and again. The output `X_data` contains as many images as defined by `batch_size`. The output `y_data` contains the measurements for these images (only the steering angle in this example).

The parameter `valid_percentage` defines how much of the dataset is used for validation leaving the rest for training. In this example 20 percent of the complete dataset is reservered for validation.

Separate generators are created for training (`train_generator`), validation (`valid_generator`) and displaying  (`display_generator`) images. The `display_generator` is used to loop through all images for visualization only.

```python
def get_data_generator(imagefiles, measurements, bmustflip, batch_size):
# ...
# Retrieve all input data
# ...
# Inputs
# ...
# imagefiles   : image files for training
# measurements : related measurements for training
# bmustflip    : boolean for 'image must be flipped'
# batch_size   : batch size which is returned for each next call
# ...
# Outputs via yield
# ...
# X_data : x values for training
# y_data : y values for training
    
    # define constants
    measurement_range = [0] # only take the first measurement (steering angle)
    
    # loop forever so the generator never terminates
    while 1:
        
        # shuffle inputs each time this loop restarts
        shuffle(imagefiles, measurements, bmustflip)
        
        # loop through all batches
        for offset in range(0, len(imagefiles), batch_size):
            
            # get batch input data
            batch_imagefiles = imagefiles[offset:(offset + batch_size)]
            batch_measurements = [measurement[index] for measurement in measurements[offset:(offset + batch_size)] \
                                  for index in measurement_range]
            batch_bmustflip = bmustflip[offset:(offset + batch_size)]
            batch_bmustautoencode = bmustautoencode[offset:(offset + batch_size)]
            
            # loop through all images
            batch_images = []
            for batch_imagefile, batch_bmustflipit in zip(batch_imagefiles, batch_bmustflip):
                
                # read image
                image = read_image(batch_imagefile)
                
                # flip image if required
                if batch_bmustflipit:
                    image = np.fliplr(image)
                    
                # add image to output
                batch_images.append(image)
            
            # calculate outputs
            X_data = np.array(batch_images)
            y_data = np.array(batch_measurements)
            
            yield X_data, y_data

# define constants
subfolder = '../../GD_GitHubData/behavioral-cloning-data'
valid_percentage = 0.2
steeroffset = 0.05
batch_size = 32
bdisplay = True

# retrieve input data
imagefiles, measurements, bmustflip, ysize, xsize = get_data(subfolder, steeroffset, bdisplay)

# need to shuffle and split into training and validation data
imagefiles_train, imagefiles_valid, measurements_train, measurements_valid, bmustflip_train, bmustflip_valid, = train_test_split(imagefiles, measurements, bmustflip, test_size = valid_percentage)
train_size = len(imagefiles_train)
valid_size = len(imagefiles_valid)
display_size = len(imagefiles)

# define data generators to retrieve batches for training and validation
train_generator = get_data_generator(imagefiles_train, measurements_train, bmustflip_train, batch_size, [0, ysize])
valid_generator = get_data_generator(imagefiles_valid, measurements_valid, bmustflip_valid, batch_size, [0, ysize])
display_generator = get_data_generator(imagefiles, measurements, bmustflip, batch_size, [0, ysize])
```

### 2. Flexible model definition

The model can be defined as sequence of convolutional layers followed by a sequence of fully connected layers. In order to define the layers the following classes have been defined:

1. `ConvLayer` for a single convolutional layer
1. `FullLayer` for a single fully connected layer
1. `ModelParameters` for the model itself

A convolutional layer is defined by the number of features (`features`), the filter size (`filter_size`), the number of strides in each dimension (`strides`) as well as whether or not *max pooling* is used (`busepooling`).

A fully connected layer is defined by the number of features (`features`) and the percentage of connections that should be kept (`keep_percentage`). If `keep_percentage` is less than 1, a *dropout* layer is added.

The model takes a list of convolutional layers (`conv_layers`: list of objects of class `ConvLayer`), fully connected layers (`full_layers`: list of objects of class `FullLayer`) and a `regularizer` object as inputs. The `regularizer` can either be `None` or a `keras.regularizers` object. The `regularizer` is applied to the kernel weights of each convolutional layer.

```python
class ConvLayer:
# ...
# Parameters used to define the structure of a convolutional layer
# ...
    
    features = 1
    filter_size = (5, 5)
    strides = (2, 2)
    busepooling = False
    
    def __init__(self, features, filter_size, strides, busepooling):
        self.features = features
        self.filter_size = filter_size
        self.strides = (1, 1) if (strides == None) else strides
        self.busepooling = busepooling

class FullLayer:
# ...
# Parameters used to define the structure of a full layer
# ...
    
    features = 1
    keep_percentage = 1
    
    def __init__(self, features, keep_percentage):
        self.features = features
        self.keep_percentage = keep_percentage

class ModelParameters:
# ...
# Parameters used to define the structure of the convolutional neural network
# ...
    
    conv_layers = []
    full_layers = []
    regularizer = None
    
    def __init__(self, conv_layers, full_layers, regularizer):
        self.conv_layers = conv_layers
        self.full_layers = full_layers
        self.regularizer = regularizer
```

The following parameters define a model with 5 convolutional layers and 4 fully connected layers that was suggested by the Udacity Self-Driving Car Engineer class. The 5 convolutional layers work well with an image of the size 320x65 pixels. The 4 fully connected layers are capable of reducing 2112 features to a single steering angle as output. The model size is large enough to accomodate L2 regularization in the convolutional layers and 50 percent dropout in the first 3 fully connected layers.

```python
# define constants
iternames = []
sMPs = []

# define parameters for configuration 0
iternames.append('c5_d4_wd')
conv_layers = []
conv_layers.append(ConvLayer(features = 24, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 36, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 48, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 64, filter_size = (3, 3), strides = None, busepooling = False))
conv_layers.append(ConvLayer(features = 64, filter_size = (3, 3), strides = None, busepooling = False))
full_layers = []
full_layers.append(FullLayer(features = 100, keep_percentage = 0.5))
full_layers.append(FullLayer(features = 50, keep_percentage = 0.5))
full_layers.append(FullLayer(features = 10, keep_percentage = 0.5))
full_layers.append(FullLayer(features = 1, keep_percentage = 1))
sMPs.append(ModelParameters(conv_layers = conv_layers.copy(), full_layers = full_layers.copy(), \
                            regularizer = regularizers.l2(0.01)))
```

The function `train_model` is used to create and train a model.

TEXT

```python
def train_model(itername, train_generator, train_size, valid_generator, valid_size, display_generator, display_size, \
                batch_size, yimagerange, ysize, xsize, epochs, modelfilename, modelfileext, modellayoutpicfilename, \
                modellayoutpicfileext, sMP, bdisplay = False, bdebug = False):
# ...
# Train model
# ...
# Inputs
# ...
# itername               : name of training iteration
# train_generator        : variable pointing to function that retrieves next values from training generator
# train_size             : total number of training data sets
# valid_generator        : variable pointing to function that retrieves next values from validation generator
# valid_size             : total number of validation data sets
# display_generator      : variable pointing to function that retrieves next values from display generator
# display_size           : total number of display data sets
# batch_size             : batch size
# yimagerange            : range of pixels used from source images in vertical direction
# ysize                  : source image height in pixels
# xsize                  : source image width in pixels
# epochs                 : number of epochs
# modelfilename          : file name in which model will be saved
# modelfileext           : file extension for file in which model will be saved
# modellayoutpicfilename : picture file in which model layout will be stored
# modellayoutpicfileext  : file extension for picture file in which model layout will be stored
# sMP                    : object containing model parameters that define the model layout
# bdisplay               : boolean for 'display information'
# bdebug                 : boolean for 'debug generator'
    
    # define Keras model
    model = Sequential()
    
    # define Keras input adjustments
    model.add(Cropping2D(cropping = ((yimagerange[0], (ysize - yimagerange[1])), (0, 0)), input_shape = (ysize, xsize, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, \
                     input_shape = (3, (ysize - (yimagerange[0] + (ysize - yimagerange[1]))), xsize)))
    
    # define Keras convolutional layers
    for conv_layer in sMP.conv_layers:
        model.add(Conv2D(conv_layer.features, conv_layer.filter_size[0], conv_layer.filter_size[1], \
                         subsample = conv_layer.strides, activation = "relu", kernel_regularizer = sMP.regularizer))
        if conv_layer.busepooling: model.add(MaxPooling2D())
    
    # define Keras dense layers
    model.add(Flatten())
    for full_layer in sMP.full_layers:
        if (full_layer.keep_percentage < 1): model.add(Dropout(full_layer.keep_percentage))
        model.add(Dense(full_layer.features))
    
    # print the layout of the model
    plot_model(model, to_file = (modellayoutpicfilename + '_' + itername + modellayoutpicfileext), show_shapes = True, \
               show_layer_names = True)
    model.summary()
    
    # generate model
    model.compile(loss = 'mse', optimizer = 'adam')
    
    # train model
    history_object = model.fit_generator(train_generator, samples_per_epoch = np.int(train_size // batch_size), \
                                         validation_data = valid_generator, \
                                         nb_val_samples = np.int(valid_size // batch_size), nb_epoch = epochs, verbose = 1)
                                         
    # save trained model
    model.save((modelfilename + '_' + itername + modelfileext))
```

The exact layer sizes and number of parameters are calculated by Keras as follows. In addition, the below visual representation was created using the `plot_model` function from `keras.utils`.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________
```

<img src="docu_images/03_01_model_c5_d4_wd.png" width="30%">

subfolder = '../../GD_GitHubData/behavioral-cloning-data'
yimagerange = [70, 135]
max_train_size = 9999999999 # 256
max_valid_size = 9999999999 # 256
max_display_size = 10
valid_percentage = 0.2
steeroffset = 0.05 # 0.2
batch_size = 32 # 256
epochs = 3
modelfilename = 'model'
modelfileext = '.h5'
modellayoutpicfilename = 'model'
modellayoutpicfileext = '.png'
bdisplay = True
bdebug = False


## 4. Selecting the model architecture and hyperparameters

### 1. Basic structure of the model

### 2. Considered model variations

### 3. Hyperparameter tuning

## 5. Evaluating the model bahavior

### 1. What does underfitting and overfitting mean in this example?

### 2. Which model architecture and what parameters worked best?

## 6. Discussion










# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
