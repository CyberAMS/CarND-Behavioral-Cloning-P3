import glob
import os
import csv
#import cv2
import matplotlib.image as mpimg
import numpy as np
#import random
import sklearn

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
    #image = cv2.imread(imagefile)
    image = mpimg.imread(imagefile)
    
    # convert color to RGB
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def get_data_generator(imagefiles, measurements, bmustflip, batch_size = 32):
# ...
# Retrieve all input data
# ...
# Inputs
# ...
# imagefiles   : image files for training
# measurements : related measurements for training
# bmustflip    : boolean for 'image must be flipped'
# ...
# Outputs
# ...
# X_train : x values for training
# y_train : y values for training
    
    # define constants
    yimagerange = [70, 140]
    
    # loop forever so the generator never terminates
    while 1:
        
        # shuffle inputs together
        #inputs = list(zip(imagefiles, measurements))
        #random.shuffle(inputs)
        #imagefiles, measurements = zip(*inputs)
        sklearn.utils.shuffle(imagefiles, measurements)
        
        for offset in range(0, len(imagefiles), batch_size):
            
            batch_imagefiles = imagefiles[offset:(offset + batch_size)]
            batch_measurements = measurements[offset:(offset + batch_size)]
            batch_bmustflip = measurements[offset:(offset + batch_size)]
            
            # loop through all images
            batch_images = []
            for batch_imagefile in batch_imagefiles:
                
                # read image
                image = read_image(batch_imagefile)
                
                # crop image
                image = image[0, :, :]
                
                # flip image if required
                if 
                batch_images.append(image)
            
            # calculate outputs
            X_train = np.array(batch_images)
            y_train = np.array(batch_measurements)
            
            sklearn.utils.shuffle(X_train, y_train)
            print(X_train.shape)
            print(y_train.shape)
            
            yield X_train, y_train

def get_data(subfolder):
# ...
# Retrieve all input data
# ...
# Inputs
# ...
# subfolder : path to folder with input data
# ...
# Outputs
# ...
# imagefiles   : image files for training
# measurements : related measurements for training
# bmustflip    : boolean for 'image must be flipped'

    # define constants
    csvmask = '*.csv'
    delimiter = ','
    drivefilename = '_driving_log.csv'
    imagefolderpostfix = '_IMG'
    steeroffset = 0.02
    
    # initialize outputs
    imagefiles = []
    measurements = []
    bmustflip = []
    
    # get path to all csv files
    files = glob.glob(os.path.join(subfolder, csvmask))
    
    # loop through all csv files
    for file in files:
        
        # separate path from file name
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
    
    return imagefiles, measurements, bmustflip

# define constants
subfolder = '../../GD_GitHubData/behavioral-cloning-data'

# commands to execute if this file is called
if __name__ == "__main__":
    
    # retrieve input data
    imagefiles, measurements, bmustflip = get_data(subfolder)
    print(len(imagefiles))
    print(len(measurements))
    
    # need to shuffle and split into training and validation data
    sklearn.utils.shuffle(imagefiles, measurements, bmustflip)
    # code to split
    
    # retrieve batch for training
    X_train, y_train = get_data_generator(imagefiles, measurements, bmustflip, batch_size = 32)
    print(X_train.shape)
    print(y_train.shape)