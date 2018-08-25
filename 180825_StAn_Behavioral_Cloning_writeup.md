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

[image1]: docu_images/01_01_left_2018_08_18_06_11_22_467.jpg {: width="200px"}
[image2]: docu_images/01_02_center_2018_08_18_06_11_22_467.jpg {: width="200px"}
[image3]: docu_images/01_03_right_2018_08_18_06_11_22_467.jpg {: width="200px"}

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

In order to run the *simulator* in conjunction with a provided `drive.py` python script in *Autonomous Mode*, I also needed to manually install the `socketio` package in the `tensorflow_GPU` environment.

### 2. Hardware considerations

I strongly recommend using a GPU for running the *simulator* in conjunction with the `drive.py` python script in *Autonomous Mode*. I am using an *NVIDIA GeForce GTX 1060 6 GB* graphics card for this. If I run the *simulator* without a GPU on an *Intel i7* platform with the fastest low resolution settings, I will experience response time issues between the *simulator* and the python `drive.py` script. The python `drive.py` script itself does not require a GPU.

In order to read images quickly from the hard disk during training, I highly recommend to use a fast *M.2 Solid-State Drive*. Compared to running it on a fast network attached storage this sped up the training process from many hours to a few minutes.

## 2. Capturing the behavioral training data

### 1. Basic strategy

Recording data in the *simulator* will save a sequence of images (center view, left view, right view) as well as the corresponding  measurements for steering angle, throttle, braking and speed.

The goal of this project is to have a car drive in the center of the provided track. Hence, collecting data of good center driving is essential. This will train the model on which steering angle to choose when seeing a specific image in the center view. The left and right views can be used to augment the training data with views that require a slight steering adjustment to get back to the center line. In the following the slight steering adjustment is defined by the parameter `steeroffset` (value greater or equal to 0). The following images show a left, center and right view of the same training data set. The steering angle `steering_angle` is a value between -1 and 1. A slightly negative value demands a slight steering to the left. The car uses maximum throttle for `throttle` (value between 0 and 1), no braking for `braking` (value between 0 and 1) and drives with a maximum speed `speed` (value between 0 and 30-ish).

![alt text][image1]![alt text][image2]![alt text][image3]

```
left view: steering_angle = -0.03759398 + steeroffset
center view: steering_angle = -0.03759398
right view: steering_angle = -0.03759398 - steeroffset
throttle = 1
braking = 0
speed = 30.19014
```

### 2. Pre-processing of recovery data

Although the center line training includes center line driving and a soft version for recovery, the car will eventually drift away from the center line due to either varying initial conditions, changing environments or prediction errors. Therefore, a stronger version for recover is needed when the car gets close to the boundaries of the driveable track.

We need to train the model to steer away from the track boundary if the car gets too close to it. We can record the necessary steering angle based on the behavior of a real driver by driving on the track and continuously weaving from left to right. Then we pre-process this recovery data to only consider images that steer the car away from the boundary. We don't want to train the model to steer towards the track boundary. We also don't want to train the model to cross over the center line with a steering angle that moves the car away from the center line like we do during the weaving events.

As the measurements are recorded in a *\*.csv* format I conveniently used *Microsoft Excel* to mark the rows that contain valid recovery situations during the weaving events. A valid recovery situation is determined as being part of the first third of an event when the steering wheel clearly changes from left to right steering or the other way round. The second third would be considered crossing the center line and the last third would be considered steering towards the closest boundary.

![alt text][image1]![alt text][image2]![alt text][image3]


The following table shows the columns of the *\*.csv* file and the values that they contain.

| Row number | Column D | Column E | Column F | Column G |
|------------|----------|----------|----------|----------|
| 6 | \<steering angle\> | \<throttle\> | \<braking\> | \<speed\> |

The next table shows the formulas I used to mark the valid recovery situations starting in row 6. Row 1 to 5 cannot contain valid situations. Column I is used to identify whether the new steering angle is part of a recovery (1 if yes and 0 if not) from the right (steering angle larger than the average of the last 5 events or steering angle negative) or the left (steering angle smaller than the average of the last 5 events or steering angle positive) side. Column J increases the counter for how many steps the most recent recovery already took by 1 if a recovery takes place - counting from the end to the beginning. Column K divides column J by 3 and is used by column L to count down by 1 step each row. Column M uses this information to return 1 if the current row is part of the first third of the latest recovery event and 0 otherwise.

| Row number | Column I | Column J | Column K | Column L | Column M |
|------------|----------|----------|----------|----------|----------|
| 6 | =IF(OR(AND(D6>=AVERAGE(D1:D5),D6<0),AND(D6<=AVERAGE(D1:D5),D6>0)),1,0) | =IF(I6=0,0,J7+I6) | =IF(J6=0,0,J6/3) | =IF(K6=0,0,IF(AND(K5=0,K6>0),K6,L5-1)) | =IF(L6>=1,1,0) |

Columns H to L are deleted before exporting the *Microsoft Excel* file to a *\*.csv'* file. Column M stays and is read as variable `btake` for each image.

### 3. Augmentation of behavioral training data

To further augment the training data I also drove the track in the opposite direction and applied all the pre-processing on this data as described in the section before.







I loaded the provided traffic signs data set and used basic Python operations to get an overview of the content:

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

The following diagram shows the amount of training, validation and testing images per traffic sign label in the provided data set.

![alt text][image39]

The numerics of neural networks work best if the mean of the input data is zero. I used the following equations to normalize all input data to be between -1 and 1.

```python
# normalize input data
X_train_norm = np.asarray(((X_train / 127.5) - 1), np.float32)
X_valid_norm = np.asarray(((X_valid / 127.5) - 1), np.float32)
X_test_norm = np.asarray(((X_test / 127.5) - 1), np.float32)
```

As the color of traffic signs is used very intentionally to cluster signs by their importance, I decided that my detection algorithm should not be based on gray scale images only. Therefore, I left the color channel in the input data.

I did not augment the training data set by generating additional input using transformations. This is a potential future improvement.

### 2. Visualization of the data

In order to plot a set of traffic sign images with labels I created the function `plot_traffic_signs`. I used this function to understand the picture content by looking at the training dataset in the following ways.

* 30 random images

![alt text][image1]

* 30 random images of the same random label

![alt text][image2]

* A very high contrast image for each label (selected by looking for the highest contrast in the center of the image)

![alt text][image3]

* The average image for each label (calculated by averaging the individual color channels of all images with the same label)

![alt text][image4]

## 2. Convolutional neural network architecture

### 1. Model definition

I started by adapting the `LeNet` example to take color images as input.

My first intention was to create a very flexible set of functions that defines a typical type of convolutional neural network based on a few input parameters for the graph layout. Unfortunately, I ran into the issue that I couldn't get the same result when choosing the parameters to represent the simple `LeNet` example. My attempts are documented in the functions `LeNet_adjusted_method` and `LeNet_adjusted_inlinemethod`. I suspect that the way Python transfers variables between functions and how I implemented this leads to missing links in the Tensorflow graph during execution.

I reverted back to defining all Tensorflow variables in a single function `LeNet_adjusted3`. The high level structure of the convolutional neural network is shown below as generated by this function.

```
Convolutional layer   1 : [32, 32] input dimension with depth 3 and [28, 28] output dimensions with depth 18
Convolutional layer   2 : [28, 28] input dimension with depth 18 and [20, 20] output dimensions with depth 54
Pooling layer         2 : [20, 20] input dimension with depth 54 and [10, 10] output dimensions with depth 54
Convolutional layer   3 : [10, 10] input dimension with depth 54 and [6, 6] output dimensions with depth 128
Fully connected layer 1 : 4608 input dimensions and 800 output dimensions
Fully connected layer 2 : 800 input dimensions and 84 output dimensions
Fully connected layer 3 : 84 input dimensions and 43 output dimensions
```

I designed the first convolutional layer with a filter size of 5 to detect 18 features instead of 6 like `LeNet`, because I am using 3 color channels in the input data. I intentionally skipped pooling in the first layer to keep as much detail as possible.

The second convolutional layer uses a larger filter size of 9 to detect larger features in the picture. I decided to triple the number of possible features when combining smaller features. To keep the network reasonably small, the second layer uses max pooling with a stride of 2.

The third convolutional layer again uses a filter size of 5 and transforms most of the remaining image structure into a total of 128 features. No pooling is used in the third layer as the size of the network is reasonably small.

The following three fully connected layers transform the features into class probabilities by continuously reducing the dimensions from 4608 to 800 to 84 and finally 43 - one class for each traffic sign label.

Each layer of the convolutional neural network `LeNet_adjusted3` uses *RELU* units followed by *dropout* except the last fully connected layer.

The model pipeline uses the `AdamOptimizer` from Tensorflow for training. The loss function is based on `reduce_mean` from Tensorflow using `softmax_cross_entropy_with_logits`. To further avoid overfitting, the weight matrices of the first and second convolutional layer get regularized using `l2_loss`.

The model accuracy is evaluated by calculating the average difference between the predicted labels and the one hot encoded input labels.

### 2. Hyper parameter selection and training

All layers of the model use random states as initial values with a mean of zero and standard deviation of 0.1.

My starting point for the hyperparameter selection is shown in the below code section. The variable `epochs` defines the number of training epochs. The `batch_size` defines how many inputs are used between every update of the internal parameters. I selected the learning rate `rate` smaller than the standard `AdamsOptimizer` setting expecting a smoother progression. All layers with *dropout* kept 50 percent of their connections as defined by `keep_prob`. The parameter `beta` is used as factor during regularization of the convolutional weights in the loss function.

```
# define constants
epochs = 20
batch_size = 256
rate = 0.0001
keep_prob = 0.5
beta = 0.2
```

The first thing I realized was that the training rate `rate` was selected too small. The model had problems leaving the initialization state. So I boldly moved to a value of 0.01 which led to a similar problem. I finally realized that values between 0.0005 and 0.001 gave the best training result. That's why I settled on a value slightly smaller than the default `AdamsOptimizer` setting of 0.001.

In the next step I tried to understand the relationship between the number of epochs `epochs` and the selected batch size `batch_size`. The model trained very well with batch sizes between 32 and 256 and needed between 20 and 50 epochs to settle on a validation accuracy above 93 percent. As expected the smaller the batch size the less epochs were needed. I did realize though that the model performed better on random pictures from the web if I used larger batch sizes, probably because the model is considering a wider variety of images before making any adjustments to the internal model parameters. For example, if the batch size is smaller than the number of different traffic sign classes, the model tries to learn a subset of traffic signs during a single batch instead of considering a wider variety. With more and more epochs this effect is of lesser importance.

The *dropout* parameter `keep_prob` applies to all layers except the last one. A value of 0.5 did not allow to reach a validation accuracy significantly above 90 percent. Instead of removing the *dropout* from some layers I decided to pick a smaller dropout percentage. Selecting larger values for the regularization parameter `beta` made it also difficult to reach a validation accuracy significantly above 90 percent. Hence, I decided to only apply a little regularization to the model.

I finally settled on the following hyperparameters to train the model which I used in the following sections.

```
# define constants
epochs = 50
batch_size = 128
rate = 0.0005
keep_prob = 0.7
beta = 0.1
```

The training progress is shown in the following diagram. After 50 epochs the model achieved an accuracy of 96.0 percent.

![alt text][image5]

I did not need to adjust my model architecture during training, because it worked very well right from the beginning. Out of curiosity I changed to less features per convolutional layer and even tried the original `LeNet` settings with only two convolutional layers (although keeping 3 channels for all colors as inpout). The more features and layers I removed the more difficult it was to tune the hyperparameters to achieve an accuracy above 90 percent.

### 3. Test of model performance after training

The accuracy on the test data set is 94.3 percent.

## 3. Predictions with the trained model

### 1. Test images from the web

To further test whether the model is good in not only predicting the images on which it has been trained, 6 images of German traffic signs have been found using [Google's image search](https://images.google.com/). I created a function `load_images` to load such a sequence of images.

![alt text][image6]

The function `predict_image_labels` uses the previously described model to predict the labels of these untrained images. The function `check_accuracy` provides a quick check whether the untrained images are accurately predicted. A more thorough check is defined by the function `evaluate_top_k` which is used in the following.

The model accurately predicts each of these untrained traffic signs as shown in the following picture sequence. The top 5 predictions are shown using the average image for each of these labels. The bar charts show the *softmax* probability for each prediction.

The *Speed limit (80 km/h)* traffic sign can easily be confused with the *Speed limit (30 km/h)* traffic sign. For example, if the left side of the *8* was exposed to bad lighting in the test image, the prediction would not be as clear as shown below.

![alt text][image7]![alt text][image8]

Both *Road work* traffic sign examples are predicted very well. If road work symbol in the center wouldn't be as clearly visible as in these test images from the web, the prediction would be much harder and any of the red triangle pictures would be a good guess.

![alt text][image9]![alt text][image10]
![alt text][image11]![alt text][image12]

The *Keep right* traffic sign is very distinctive and hence gets detected very well. It is interesting that the other predictions in the top 5 are not blue colored traffic signs. It seems like color is not the highest distinguishing factor in the given model and general shapes are more important. Therefore, the shape of traffic signs should not be distorted in the test images to ensure a proper prediction.

![alt text][image13]![alt text][image14]

The *No passing* traffic sign test image from the web is angled in a way that the red car on the left nearly looks like a truck pictogram from a *No passing for vehicles over 3.5 metric tons* traffic sign. Indeed this is the model's second best guess.

![alt text][image15]![alt text][image16]

And finally the *Stop* traffic sign image from the web is so distinctive that even worse trained model versions picked it up accurately.

![alt text][image17]![alt text][image18]

The test images from the web have all been predicted accurately which exceeds the accuracy of the test data set. On the one hand looking at only 6 test images from the web is statistically not relevant. On the other hand using Google's search to find traffic sign images can be assumed to provide nice pictures of traffic signs. Even when I tried searching for *bad traffic sign scenery* in German language, most of the traffic sign images were very nice pictures - or snow covered sceneries with piles of snow around a traffic sign - and even a human cannot predict anything in these images. But I didn't give up and had some fun with other traffic signs further below.

### 2. Exploring the inside of the model

In order to visualize the weights in the convolutional layers of my neural network, I evaluated the model with the average *Stop* sign image from the training data set using the function `outputFeatureMap`.

![alt text][image19]

The first convolutional layer uses 3 color channels as input. Therefore, I chose to visualize the weights of each feature map using an *RGB* color image. The 18 feature maps of this layer clearly show that they are distinct by having different average color tones. The images on the left are more green-ish while the ones on the right show more blue and red color tones. Also, the area in which they emphasize on specific color inputs is very different. Especially the images in the center seem to emphasize on diagonal directions.

![alt text][image20]

The second convolutional layer has 54 feature maps for each of the 18 input channels. The following picture only shows the first 250 feature maps as grayscale images (a little more than 25 percent of all feature maps in this layer). Due to the larger size of the filter in the second layer, some of the features are very detailed while others focus on more general patterns.

![alt text][image21]

The third and final convolutional layer has 128 feature maps for each of the 54 input channels. The following picture only shows the first 250 feature maps as grayscale images (less than 4 percent of all feature maps in this layer). The filter size is again smaller and hence the individual features in these feature maps are coarser.

![alt text][image22]

### 3. Fun with unknown traffic signs

The real fun with convolutional neural networks starts when we use them for something that they have not been trained for. What would they predict when they see US instead of German traffic signs? Let's try it!

For the *Intersection* sign the model probably focuses on the large vertical black line in the center and thinks it might be a *General Caution* sign.

![alt text][image23]![alt text][image24]

The *Pedestrian Crossing* sign clearly puts the model out of its comfort zone and it cannot predict anything with a high probability.

![alt text][image25]![alt text][image26]

I think it is facinating that my model accurately predicts a *Yellow Stop* sign to be a *Stop* sign with pretty high probability although my model also takes into account the color.

![alt text][image27]![alt text][image28]

Now here it is: Who would have guessed that the US *Traffic Light* sign is close to the German *Traffic Light* sign? Well, I didn't and so does the model. It doesn't even consider it as part of the top 5 predictions.

![alt text][image29]![alt text][image30]

And it gets better: The *Right Turn Ahead* signs are pretty close besides being edgy versus round and the model nails it.

![alt text][image31]![alt text][image32]

Well, and here is the *Right Turn* sign that is really for away from anything in the training data set. Similar to the first sign in this fun sequence, the model probably focuses on the vertical black line in the middle and predicts a *General Caution* sign. Well, it's always good to be cautious with your predictions - or not?

![alt text][image33]![alt text][image34]

### 4. Searching within larger pictures

In real life we probably don't have the luxury of predicting mug shots of traffic signs. We have to find them in a larger image. And now they can be larger or smaller. I created a little algorithm in the function `check_large_images` that scans a larger image with different scaling levels and tries to find traffic signs of any size in any position.

The below pictures show the original image, a picture in which I marked all areas in which potential traffic signs have been predicted and finally a picture in which only the top predictions have been marked. The predicted areas have a yellow outline for the most likely prediction and a more and more black outline for the lesser likely predictions. As expected the most likely predictions for traffic signs are in the area of the actual traffic signs in the larger image.

![alt text][image35]![alt text][image36]![alt text][image37]

But which specific traffic signs does the model predict? It accurately lists the *Yield* and *Roundabout Mandatory* signs. Somehow it gets confused with the *Roundabout Mandatory* sign and detects many possible individual directions with even higher probability.

![alt text][image38]

## 4. Discussion

Most images used to test my model are nice frontal shots of sunny day traffic signs. Looking at some of the results makes it very obvious that traffic signs that are covered by obstacles or snow or images taken at night during rain would challenge my model extremely.









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
