import glob
import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    steeroffset = 0.02
    
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

def train_model(data_generator, datasets, batch_size, bdisplay = False):
# ...
# Train model
# ...
# Inputs
# ...
# data_generator : variable pointing to function that retrieves next values from data generator
# datasets       : total number of data sets
# batch_size     : batch size
# bdisplay  : boolean for 'display information'
    
    # loop through all batches
    for dataset in range(0, datasets, batch_size):
        
        # get batch data
        X_train, y_train = next(data_generator)
        
        # display information
        if bdisplay:
            print('Training data set', dataset, 'of', datasets, ':', X_train.shape, '=>', y_train.shape)
            plt.imshow(X_train[0])
            plt.show()
            
# define constants
subfolder = '../../GD_GitHubData/behavioral-cloning-data'
valid_percentage = 0.1
batch_size = 32
bdisplay = True

# commands to execute if this file is called
if __name__ == "__main__":
    
    # retrieve input data
    imagefiles, measurements, bmustflip = get_data(subfolder, bdisplay)
    
    # need to shuffle and split into training and validation data
    imagefiles_train, imagefiles_valid, measurements_train, measurements_valid, bmustflip_train, bmustflip_valid = train_test_split(imagefiles, measurements, bmustflip, test_size = valid_percentage)
    
    # define data generator to retrieve batch for training
    data_generator = get_data_generator(imagefiles_train, measurements_train, bmustflip_train, batch_size)
    
    # train model
    #train_model(data_generator, len(imagefiles_train), batch_size, bdisplay = bdisplay)
    train_model(data_generator, 50, batch_size, bdisplay = bdisplay)