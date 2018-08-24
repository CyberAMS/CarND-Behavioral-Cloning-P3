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

def correct_path(filepath, subfolder, imagefoldername):
# ...
# Correct path of file to new subfolder
# ...
# Inputs
# ...
# filepath        : file name with old path
# subfolder       : new path to file
# imagefoldername : name of image folder
# ...
# Outputs
# ...
# newfilepath : file name with new path
    
    # separate old path from file name
    pathname, filename = os.path.split(filepath)
    
    # add new path to file name
    newfilepath = os.path.join(subfolder, imagefoldername, filename)

    return newfilepath

def read_image(imagefile):
# ...
# Read image from file
# ...
# Inputs
# ...
# imagefile : image file name
# ...
# Outputs
# ...
# image : RGB image data
    
    # read source file
    image = cv2.imread(imagefile)
    
    # convert color to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def get_data(subfolder, steeroffset, bdisplay = False):
# ...
# Retrieve all input data
# ...
# Inputs
# ...
# subfolder   : path to folder with input data
# steeroffset : steering offset for left and right images
# bdisplay    : boolean for 'display information'
# ...
# Outputs
# ...
# imagefiles      : image files for training
# measurements    : related measurements for training
# bmustflip       : boolean for 'image must be flipped'
# bmustautoencode : boolean for 'image must be autoencoded'
# ysize           : height of images
# xsize           : width of images

    # define constants
    csvmasks = ('track1_center*.csv', 'track1_counter*.csv', 'track1_weave*.csv', 'track1_ceave*.csv', 'track1_meave*.csv', \
                'track1_deave*.csv')
    #csvmasks = ('track1_center*.csv', 'track1_counter*.csv', 'track1_weave*.csv', 'track1_ceave*.csv')
    delimiter = ','
    drivefilename = '_driving_log.csv'
    imagefolderpostfix = '_IMG'
    aeoffset = steeroffset
    
    # display information
    if bdisplay:
        print('Finding input data files...')
        
    # initialize outputs
    imagefiles = []
    measurements = []
    bmustflip = []
    bmustautoencode = []
    
    # get paths to all csv files
    files = []
    for csvmask in csvmasks:
        files.append(glob.glob(os.path.join(subfolder, csvmask)))
    files = [file for sublist in files for file in sublist]
    
    # loop through all csv files
    for file in files:
        
        # display information
        if bdisplay:
            print(file)
        
        # determine image folder name based on csv file name
        path, name = os.path.split(file)
        prefix = name[0:(len(name) - len(drivefilename))]
        imagefoldername = prefix + imagefolderpostfix
        
        # read csv file content
        lines = []
        with open(file) as csvfile:
            reader = csv.reader(csvfile, delimiter = delimiter)
            for line in reader:
                lines.append(line)
        
        # parse content
        for line in lines:
            
            # decipher line
            centerfile = correct_path(line[0], subfolder, imagefoldername)
            leftfile = correct_path(line[1], subfolder, imagefoldername)
            rightfile = correct_path(line[2], subfolder, imagefoldername)
            angle = np.float_(line[3])
            throttle = np.float_(line[4])
            brake = np.float_(line[5])
            speed = np.float_(line[6])
            btake = np.bool_(line[7])
            
            # check whether data point should be considered and then augment data set
            if btake:
                
                # add center image to output
                imagefiles.append(centerfile)
                measurements.append([angle, throttle, brake, speed])
                bmustflip.append(False)
                bmustautoencode.append(False)
                
                # add flipped center image to output
                imagefiles.append(centerfile)
                measurements.append([-angle, throttle, brake, speed])
                bmustflip.append(True)
                bmustautoencode.append(False)
                
                # add left image to output
                imagefiles.append(leftfile)
                measurements.append([(angle + steeroffset), throttle, brake, speed])
                bmustflip.append(False)
                bmustautoencode.append(False)
                
                # add flipped left image to output
                imagefiles.append(leftfile)
                measurements.append([-(angle + steeroffset), throttle, brake, speed])
                bmustflip.append(True)
                bmustautoencode.append(False)
                
                # add right image to output
                imagefiles.append(rightfile)
                measurements.append([(angle - steeroffset), throttle, brake, speed])
                bmustflip.append(False)
                bmustautoencode.append(False)
                
                # add flipped right image to output
                imagefiles.append(rightfile)
                measurements.append([-(angle - steeroffset), throttle, brake, speed])
                bmustflip.append(True)
                bmustautoencode.append(False)
                
                # potential augmentation with autoencoder and offset to left and right
                # would need to adjust steering angle, too (linear increase from center to sides and
                # extreme increase if already on the sides)
                #imagefiles.append(leftfile) # shift this image to be further left
                #measurements.append([-(angle - steeroffset - aeoffset), throttle, brake, speed])
                #bmustflip.append(False)
                #bmustautoencode.append(True)
                # do this for right image and flipped left and right images, too

    
    # get size of images
    image = cv2.imread(imagefiles[0])
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    # display information
    if bdisplay:
        print('Number of image files:', len(imagefiles), '; number of measurements:', len(measurements), '; must flip:', \
              np.sum(np.array(bmustflip) == True), '; size:', xsize, 'x', ysize)
        
    return imagefiles, measurements, bmustflip, bmustautoencode, ysize, xsize

def auto_encoder(imagefiles, bmustflip, ysize, xsize):
# ...
# Create and train auto-encoder model for input image augmentation
# ...
# Outputs
# ...
# auto_encoder_model : trained auto-encoder model
    
    # define Keras auto-encoder model
    auto_encoder_model = Sequential()
    
    return auto_encoder_model

def get_data_generator(imagefiles, measurements, bmustflip, bmustautoencode, auto_encoder_model, batch_size, yimagerange):
# ...
# Retrieve all input data
# ...
# Inputs
# ...
# imagefiles         : image files for training
# measurements       : related measurements for training
# bmustflip          : boolean for 'image must be flipped'
# bmustautoencode    : boolean for 'image must be autoencoded'
# auto_encoder_model : trained auto-encoder model
# batch_size         : batch size which is returned for each next call
# yimagerange        : range of pixels used from source images in vertical direction
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
            for batch_imagefile, batch_bmustflipit, batch_bmustautoencodeit in zip(batch_imagefiles, batch_bmustflip, \
                                                                                   batch_bmustautoencode):
                
                # read image
                image = read_image(batch_imagefile)
                
                # auto-encode image if required
                if batch_bmustautoencodeit:
                    print('Not implemented yet!') # auto-encode image using auto_encoder_model
                
                # crop image
                image = image[yimagerange[0]:yimagerange[1], :, :]
                
                # flip image if required
                if batch_bmustflipit:
                    image = np.fliplr(image)
                    
                # add image to output
                batch_images.append(image)
            
            # calculate outputs
            X_data = np.array(batch_images)
            y_data = np.array(batch_measurements)
            
            yield X_data, y_data

def get_and_display_generator_data(data_generator, dataset, data_size, bdisplay):
# ...
# Train model
# ...
# Inputs
# ...
# data_generator : variable pointing to function that retrieves next values from data generator
# dataset        : current number of data set
# data_size      : total number of data sets
# bdisplay       : boolean for 'display information'

    # get batch data
    X_data, y_data = next(data_generator)
    
    # display information
    if bdisplay:
        
        # display all data sets
        for idx, (X, y) in enumerate(zip(X_data, y_data)):
            print('Data set', dataset, 'of', data_size, 'element', idx, ':', X_data.shape, '=>', y_data.shape)
            if isinstance(y, np.ndarray):
                print('   Angle:', '{:07.4f}'.format(y[0]) if (len(y) >= 0) else False, 'Throttle:', '{:07.4f}'.format(y[1]) \
                      if (len(y) >= 1) else False, 'Braking:', '{:07.4f}'.format([2]) if (len(y) >= 2) else False, 'Speed:', \
                      '{:07.4f}'.format([3]) if (len(y) >= 3) else False)
            else:
                print('   Angle:', '{:07.4f}'.format(y))
            plt.imshow(X)
            plt.show()

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
    
    # display information
    if bdisplay:
        
        # display number of batches
        print('= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =')
        print('Configuration:', itername)
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('Batch size:', batch_size)
    
    # debug generator
    if bdebug:
        
        # loop through all training data
        for display_dataset in range(0, display_size, batch_size):
            
            # get and display training data
            get_and_display_generator_data(display_generator, display_dataset, display_size, bdisplay)
            
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
    
    # display information
    if bdisplay:
        
        # print the keys contained in the history object
        #print(history_object.history.keys())
        
        # plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc = 'upper right')
        plt.show()
    
    # save trained model
    model.save((modelfilename + '_' + itername + modelfileext))
    
# define constants
iternames = []
subfolder = '../../GD_GitHubData/behavioral-cloning-data'
yimagerange = [70, 135]
max_train_size = 9999999999 # 32
max_valid_size = 9999999999 # 32
max_display_size = 10
valid_percentage = 0.2
steeroffset = 0.05 # 0.4 # 0.2
batch_size = 32 # 256
epochs = 3
modelfilename = 'model'
modelfileext = '.h5'
modellayoutpicfilename = 'model'
modellayoutpicfileext = '.png'
bdisplay = True
bdebug = False
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

# define parameters for configuration 1
iternames.append('c5_d4_nd')
conv_layers = []
conv_layers.append(ConvLayer(features = 24, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 36, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 48, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 64, filter_size = (3, 3), strides = None, busepooling = False))
conv_layers.append(ConvLayer(features = 64, filter_size = (3, 3), strides = None, busepooling = False))
full_layers = []
full_layers.append(FullLayer(features = 100, keep_percentage = 1))
full_layers.append(FullLayer(features = 50, keep_percentage = 1))
full_layers.append(FullLayer(features = 10, keep_percentage = 1))
full_layers.append(FullLayer(features = 1, keep_percentage = 1))
sMPs.append(ModelParameters(conv_layers = conv_layers.copy(), full_layers = full_layers.copy(), \
                            regularizer = regularizers.l2(0.01)))

# define parameters for configuration 2
iternames.append('c2_d3_wd')
conv_layers = []
conv_layers.append(ConvLayer(features = 24, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 36, filter_size = (5, 5), strides = (2, 2), busepooling = False))
full_layers = []
full_layers.append(FullLayer(features = 100, keep_percentage = 0.5))
full_layers.append(FullLayer(features = 10, keep_percentage = 0.5))
full_layers.append(FullLayer(features = 1, keep_percentage = 1))
sMPs.append(ModelParameters(conv_layers = conv_layers.copy(), full_layers = full_layers.copy(), \
                            regularizer = regularizers.l2(0.01)))

# define parameters for configuration 3
iternames.append('c2_d3_nd')
conv_layers = []
conv_layers.append(ConvLayer(features = 24, filter_size = (5, 5), strides = (2, 2), busepooling = False))
conv_layers.append(ConvLayer(features = 36, filter_size = (5, 5), strides = (2, 2), busepooling = False))
full_layers = []
full_layers.append(FullLayer(features = 100, keep_percentage = 1))
full_layers.append(FullLayer(features = 10, keep_percentage = 1))
full_layers.append(FullLayer(features = 1, keep_percentage = 1))
sMPs.append(ModelParameters(conv_layers = conv_layers.copy(), full_layers = full_layers.copy(), \
                            regularizer = regularizers.l2(0.01)))

# commands to execute if this file is called
if __name__ == "__main__":
    
    # retrieve input data
    imagefiles, measurements, bmustflip, bmustautoencode, ysize, xsize = get_data(subfolder, steeroffset, bdisplay)
    
    # create auto-encoder if necessary
    auto_encoder_model = None # auto_encoder(imagefiles, bmustflip, ysize, xsize)
    
    # need to shuffle and split into training and validation data
    imagefiles_train, imagefiles_valid, measurements_train, measurements_valid, bmustflip_train, bmustflip_valid, bmustautoencode_train, bmustautoencode_valid = train_test_split(imagefiles, measurements, bmustflip, bmustautoencode, test_size = valid_percentage)
    train_size = len(imagefiles_train)
    valid_size = len(imagefiles_valid)
    display_size = len(imagefiles)
    
    # define data generators to retrieve batches for training and validation
    train_generator = get_data_generator(imagefiles_train, measurements_train, bmustflip_train, bmustautoencode_train, \
                                         auto_encoder_model, batch_size, [0, ysize])
    valid_generator = get_data_generator(imagefiles_valid, measurements_valid, bmustflip_valid, bmustautoencode_valid, \
                                         auto_encoder_model, batch_size, [0, ysize])
    display_generator = get_data_generator(imagefiles, measurements, bmustflip, bmustautoencode, auto_encoder_model, \
                                           batch_size, [0, ysize])
    
    # train all desired model configurations
    for itername, sMP in zip(iternames, sMPs):
        train_model(itername, train_generator, np.min([train_size, max_train_size]), valid_generator, \
                    np.min([valid_size, max_valid_size]), display_generator, np.min([display_size, max_display_size]), \
                    batch_size, yimagerange, ysize, xsize, epochs, modelfilename, modelfileext, modellayoutpicfilename, \
                    modellayoutpicfileext, sMP, bdisplay = bdisplay, bdebug = bdebug)