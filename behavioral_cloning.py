# import packages
import glob
import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2
from keras.layers.pooling import MaxPooling2D

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

def get_data(subfolder, bdisplay = False):
# ...
# Retrieve all input data
# ...
# Inputs
# ...
# subfolder : path to folder with input data
# bdisplay  : boolean for 'display information'
# ...
# Outputs
# ...
# imagefiles   : image files for training
# measurements : related measurements for training
# bmustflip    : boolean for 'image must be flipped'

    # define constants
    csvmask = 'track1*.csv'
    delimiter = ','
    drivefilename = '_driving_log.csv'
    imagefolderpostfix = '_IMG'
    steeroffset = 0.05
    
    # display information
    if bdisplay:
        print('Finding input data files...')
        
    # initialize outputs
    imagefiles = []
    measurements = []
    bmustflip = []
    
    # get path to all csv files
    files = glob.glob(os.path.join(subfolder, csvmask))
    
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
            
            # check whether data point should be considered
            if btake:
                
                # add center image to output
                imagefiles.append(centerfile)
                measurements.append([angle, throttle, brake, speed])
                bmustflip.append(False)
                
                # add flipped center image to output
                imagefiles.append(centerfile)
                measurements.append([-angle, throttle, brake, speed])
                bmustflip.append(True)
                
                # add left image to output
                imagefiles.append(leftfile)
                measurements.append([(angle + steeroffset), throttle, brake, speed])
                bmustflip.append(False)
                
                # add flipped left image to output
                imagefiles.append(leftfile)
                measurements.append([-(angle + steeroffset), throttle, brake, speed])
                bmustflip.append(True)
                
                # add right image to output
                imagefiles.append(rightfile)
                measurements.append([(angle - steeroffset), throttle, brake, speed])
                bmustflip.append(False)
                
                # add flipped right image to output
                imagefiles.append(rightfile)
                measurements.append([-(angle - steeroffset), throttle, brake, speed])
                bmustflip.append(True)
    
    # display information
    if bdisplay:
        print('Number of image files:', len(imagefiles), '; number of measurements:', len(measurements), '; must flip:', len(bmustflip))
        
    return imagefiles, measurements, bmustflip

def get_data_generator(imagefiles, measurements, bmustflip, batch_size, yimagerange):
# ...
# Retrieve all input data
# ...
# Inputs
# ...
# imagefiles   : image files for training
# measurements : related measurements for training
# bmustflip    : boolean for 'image must be flipped'
# batch_size   : batch size which is returned for each next call
# yimagerange  : range of pixels used from source images in vertical direction
# ...
# Outputs via yield
# ...
# X_train : x values for training
# y_train : y values for training
    
    # define constants
    yimagerange = [70, 135]
    
    # loop forever so the generator never terminates
    while 1:
        
        # shuffle inputs each time this loop restarts
        shuffle(imagefiles, measurements, bmustflip)
        
        # loop through all batches
        for offset in range(0, len(imagefiles), batch_size):
            
            # get batch input data
            batch_imagefiles = imagefiles[offset:(offset + batch_size)]
            batch_measurements = measurements[offset:(offset + batch_size)]
            batch_bmustflip = measurements[offset:(offset + batch_size)]
            
            # loop through all images
            batch_images = []
            for batch_imagefile, batch_bmustflipit in zip(batch_imagefiles, batch_bmustflip):
                
                # read image
                image = read_image(batch_imagefile)
                
                # crop image
                image = image[yimagerange[0]:yimagerange[1], :, :]
                
                # flip image if required
                if batch_bmustflipit:
                    image = np.fliplr(image)
                    
                # add image to output
                batch_images.append(image)
            
            # calculate outputs
            X_train = np.array(batch_images)
            y_train = np.array(batch_measurements)
            
            yield X_train, y_train

def train_model(train_generator, train_datasets, valid_generator, batch_size, validation_percentage, yimagerange, ysize, xsize, epochs, modelfile, bdisplay = False):
# ...
# Train model
# ...
# Inputs
# ...
# train_generator       : variable pointing to function that retrieves next values from training generator
# valid_generator       : variable pointing to function that retrieves next values from validation generator
# train_datasets        : total number of training data sets
# batch_size            : batch size
# validation_percentage : percentage of validation data
# yimagerange           : range of pixels used from source images in vertical direction
# ysize                 : source image height in pixels
# xsize                 : source image width in pixels
# epochs                : number of epochs
# modelfile             : file name in which model will be saved
# bdisplay              : boolean for 'display information'
    
    """
    # loop through all batches
    for train_dataset in range(0, train_datasets, batch_size):
        
        # get batch data
        X_train, y_train = next(train_generator)
        
        # display information
        if bdisplay:
            print('Training data set', train_dataset, 'of', train_datasets, ':', X_train.shape, '=>', y_train.shape)
            print('   Angle:', '{:07.4f}'.format(y_train[0][0]), 'Throttle:', '{:07.4f}'.format(y_train[0][1]), 'Braking:', '{:07.4f}'.format(y_train[0][2]), 'Speed:', '{:07.4f}'.format(y_train[0][3]))
            plt.imshow(X_train[0])
            plt.show()
    """
    
    # define Keras model
    model = Sequential()
    
    # define Keras input adjustments
    model.add(Cropping2D(cropping = ((yimagerange[0], (ysize - yimagerange[1])), (0, 0)), input_shape = (3, ysize, xsize)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = ((ysize - yimagerange[0] - (ysize - yimagerange[1])), xsize, 3)))
    
    # define Keras convolutional layers
    model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu"))
    #model.add(MaxPooling2D())
    model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu"))
    #model.add(MaxPooling2D())
    model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu"))
    #model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation = "relu"))
    #model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, 3, activation = "relu"))
    #model.add(MaxPooling2D())
    
    # define Keras dense layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    # generate model
    model.compile(loss = 'mse', optimizer = 'adam')
    
    # train model
    #model.fit(X_train, y_train, validation_split = valid_percentage, shuffle = True, nb_epoch = epochs, verbose = 1)
    history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = epochs, verbose = 1)
    
    # display information
    if bdisplay:
        
        # print the keys contained in the history object
        print(history_object.history.keys())
        
        # plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
    
    # save trained model
    model.save(modelfile)
    
# define constants
subfolder = '../../GD_GitHubData/behavioral-cloning-data'
valid_percentage = 0.2
batch_size = 32
yimagerange = [70, 135]
epochs = 2
modelfile = 'model.h5'
bdisplay = True

# commands to execute if this file is called
if __name__ == "__main__":
    
    # retrieve input data
    imagefiles, measurements, bmustflip = get_data(subfolder, bdisplay)
    ysize = imagefiles.shape[1]
    xsize = imagefiles.shape[2]
    print(ysize, xsize)
    
    # need to shuffle and split into training and validation data
    imagefiles_train, imagefiles_valid, measurements_train, measurements_valid, bmustflip_train, bmustflip_valid = train_test_split(imagefiles, measurements, bmustflip, test_size = valid_percentage)
    
    # define data generators to retrieve batches for training and validation
    train_generator = get_data_generator(imagefiles_train, measurements_train, bmustflip_train, batch_size, [0, ysize])
    valid_generator = get_data_generator(imagefiles_valid, measurements_valid, bmustflip_valid, batch_size, [0, ysize])
    
    # train model
    #train_model(train_generator, valid_generator, len(imagefiles_train), batch_size, validation_percentage, yimagerange, ysize, xsize, epochs, modelfile, bdisplay = bdisplay)
    train_model(train_generator, valid_generator, 256, batch_size, validation_percentage, yimagerange, ysize, xsize, epochs, modelfile, bdisplay = bdisplay)