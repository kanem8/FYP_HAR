'''

@author: Fernando Moya Rueda

Pattern Recognition Group

Technische Universitaet Dortmund



Process the Pamap2 dataset. It selects the files, sensor channels. In addition, it normalizes

and downsamples the signal measurements.




It creates a cPickle file the three matrices (train, validation and test),

containing the sensor measurements (row for samples and columns for sensor channels) and the annotated label



The dataset can be downloaded in  

http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring



'''

import os
import numpy as np
# import six; from six.moves import cPickle as cp
import _pickle as cp
import sequence_segment as ss
import Multi_modal.generate_plots as gp


# Number of sensor channels employed in the Pamap2

NB_SENSOR_CHANNELS = 40

# File names of the files defining the PAMAP2 data.

PAMAP2_DATA_FILES = ['PAMAP2_Dataset/Protocol/subject101.dat', #0
                          'PAMAP2_Dataset/Optional/subject101.dat', #1
                          'PAMAP2_Dataset/Protocol/subject102.dat', #2
                          'PAMAP2_Dataset/Protocol/subject103.dat', #3
                          'PAMAP2_Dataset/Protocol/subject104.dat', #4
                          'PAMAP2_Dataset/Protocol/subject107.dat', #5
                          'PAMAP2_Dataset/Protocol/subject108.dat', #6
                          'PAMAP2_Dataset/Optional/subject108.dat', #7
                          'PAMAP2_Dataset/Protocol/subject109.dat', #8
                          'PAMAP2_Dataset/Optional/subject109.dat', #9

                          'PAMAP2_Dataset/Protocol/subject105.dat', #10 validation
                          'PAMAP2_Dataset/Optional/subject105.dat', #11 validation

                          'PAMAP2_Dataset/Protocol/subject106.dat', #12 test 
                          'PAMAP2_Dataset/Optional/subject106.dat', #13 test

                          ]

NORM_MAX_THRESHOLDS = [202.0, 35.5, 47.6314, 155.532, 157.76, 45.5484, 62.2598, 61.728, 21.8452,

                       13.1222, 14.2184, 137.544, 109.181, 100.543, 38.5625, 26.386, 153.582,

                       37.2936, 23.9101, 61.9328, 36.9676, 15.5171, 5.97964, 2.94183, 80.4739,

                       39.7391, 95.8415, 35.4375, 157.232, 157.293, 150.99, 61.9509, 62.0461, 

                       60.9357, 17.4204, 13.5882, 13.9617, 91.4247, 92.867, 146.651]

NORM_MIN_THRESHOLDS = [0., 0., -114.755, -104.301, -73.0384, -61.1938, -61.8086, -61.4193, -27.8044,

                       -17.8495, -14.2647, -103.941, -200.043, -163.608, 0., -29.0888, -38.1657, -57.2366,

                       -32.9627, -39.7561, -56.0108, -10.1563, -5.06858, -3.99487, -70.0627, -122.48,

                       -66.6847, 0., -155.068, -155.617, -156.179, -60.3067, -61.9064, -62.2629, -14.162,

                       -13.0401, -14.0196, -172.865, -137.908, -102.232]

def select_columns_opp(data): # data_x passed in, which has first two columns (time and labels) removed

    """Selection of the 40 columns employed in the Pamap2

    :param data: numpy integer matrix
        Sensor data (all features)

    :return: numpy integer matrix
        Selection of features

    """
    #included-excluded

    features_delete = np.arange(14, 18) # 14-17 = orientation IMU Hand (not valid data according to readme)
    features_delete = np.concatenate([features_delete, np.arange(31, 35)])
    features_delete = np.concatenate([features_delete, np.arange(48, 52)])

    return np.delete(data, features_delete, 1) # removing along x axis (columns) this time


def normalize(data, max_list, min_list):
    """Normalizes sensor channels to a range [0,1]

    :param data: numpy integer matrix
        Sensor data

    :param max_list: numpy integer array
        Array containing maximums values for every one of the 40 sensor channels

    :param min_list: numpy integer array
        Array containing minimum values for every one of the 40 sensor channels

    :return:
        Normalized sensor data
    """

    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list

    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]

    data[data > 1] = 0.99
    data[data < 0] = 0.00

    return data


def complete_HR(data):

    """Sampling rate for the heart rate is different from the other sensors. Missing
    measurements are filled

    :param data: numpy integer matrix
        Sensor data

    :return: numpy integer matrix, numpy integer array
        HR channel data
    """

    pos_NaN = np.isnan(data) # returns an array of boolean values same size as data, true where data[i] = NaN value, false if not NaN

    idx_NaN = np.where(pos_NaN == False)[0] # array with only non-NaN values

    data_no_NaN = data * 0 # initialize data_no_NaN as same size as data

    for idx in range(idx_NaN.shape[0] - 1): #for each non-NaN value
        data_no_NaN[idx_NaN[idx] : idx_NaN[idx + 1]] = data[idx_NaN[idx]]

    data_no_NaN[idx_NaN[-1] :] = data[idx_NaN[-1]]

    return data_no_NaN


def divide_x_y(data):

    """Segments each sample into time, labels and sensor channels

    :param data: numpy integer matrix
        Sensor data

    :return: numpy integer matrix, numpy integer array
        Time and labels as arrays, sensor channels as matrix
    """

    data_t = data[:, 0] # all rows, first column (column index 0) = all timestamps

    data_y = data[:, 1] # all rows, second column (column index 1) = all labels

    data_x = data[:, 2:] # all rows, all columns from index 2 on = all sensor data


    return data_t, data_x, data_y



def adjust_idx_labels(data_y):

    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function adjust the labels picking the labels
    for the protocol settings

    :param data_y: numpy integer array
        Sensor labels

    :return: numpy integer array
        Modified sensor labels
    """

    data_y[data_y == 24] = 0

    data_y[data_y == 12] = 8

    data_y[data_y == 13] = 9

    data_y[data_y == 16] = 10

    data_y[data_y == 17] = 11

    return data_y



def del_labels(data_t, data_x, data_y):

    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function deletes the nonrelevant labels

    :param data_y: numpy integer array
        Sensor labels

    :return: numpy integer array
        Modified sensor labels
    """

    # labels 0,8,9,10,11,18,19,20 not used

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]    
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 9)[0]    
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 10)[0]    
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 11)[0]    
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 18)[0]    
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 19)[0]    
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 20)[0]    
    labels_delete = np.concatenate([labels_delete, idy])

    # delete: Return a new array with sub-arrays along an axis deleted, 0 = y-axis (rows)?
    return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)


def downsampling(data_t, data_x, data_y):

    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset

    :param data_t: numpy integer array
        time array

    :param data_x: numpy integer array
        sensor recordings

    :param data_y: numpy integer array
        labels

    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_t.shape[0], 3)

    return data_t[idx], data_x[idx], data_y[idx]


def process_dataset_file(data):

    """Function defined as a pipeline to process individual Pamap2 files

    :param data: numpy integer matrix
        channel data: samples in rows and sensor channels in columns

    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into samples-channel measurements (x) and labels (y)
    """

    # Data is divided in time, sensor data and labels

    data_t, data_x, data_y =  divide_x_y(data)
    # data_t = the time column of data file
    # data_y = the label column of the data file
    # data_x = all of the sensor data of the data file (2d matrix)


    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # nonrelevant labels are deleted

    data_t, data_x, data_y = del_labels(data_t, data_x, data_y)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    
    # Labels are adjusted

    data_y = adjust_idx_labels(data_y) # adjusts the labels so we have labels 0-11 (1-7 were unchanged)
    data_y = data_y.astype(int) # casts label matrix to type integer

    # Select correct columns
    data_x = select_columns_opp(data_x) # removes the orientation data, now have 40 columns


    if data_x.shape[0] != 0:
        HR_no_NaN = complete_HR(data_x[:,0]) # passing Heart Rate column to complete_HR function (all rows, first column)
        data_x[:,0] = HR_no_NaN # all rows of column 0 (HR) set to no-NaN matrix

        data_x[np.isnan(data_x)] = 0

        #Normalizing signals per chanel to a range of [0,1]
        data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    else:
        data_x = data_x
        data_y = data_y
        data_t = data_t

        print("SIZE OF THE SEQUENCE IS CERO")

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    data_t, data_x, data_y = downsampling(data_t, data_x, data_y)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    return data_x, data_y


def generate_data(dataset, target_filename):

    """Function to read the Pamap2 raw data and process the sensor channels
    of the protocol settings

    :param dataset: string
        Path with original pamap2 folder

    :param target_filename: string
        Path of the expected file.
    """

    # NB_SENSOR_CHANNELS = 40

    # All of these arrays are just null arrays: []

    X_train = np.empty((0, NB_SENSOR_CHANNELS)) # 0 rows, 40 columns
    y_train = np.empty((0))

    X_val = np.empty((0, NB_SENSOR_CHANNELS))
    y_val = np.empty((0))

    X_test = np.empty((0, NB_SENSOR_CHANNELS))
    y_test = np.empty((0))

    counter_files = 0

    print('Processing dataset files ...')

    for filename in PAMAP2_DATA_FILES:

        if counter_files <= 9: #for training data sets

            # Train partition

            try:

                print('Train... file {0}'.format(filename))

                data = np.loadtxt(dataset + filename) # loads data in eg. D:\Fourth Year\FYP\PAMAP2_Dataset\Protocol/subject101.dat
                # returns the data as an ndarray
                print('Train... data size {}'.format(data.shape)) # shape of data container = (rows, columns)

                x, y = process_dataset_file(data)

                print(x.shape)

                print(y.shape)

                X_train = np.vstack((X_train, x)) #same as concatenate but for 2D?

                y_train = np.concatenate([y_train, y])

            except KeyError:

                print('ERROR: Did not find {0} in zip file'.format(filename))

    

        elif counter_files > 9 and  counter_files < 12: # 10 & 11: for validation data sets

            # Validation partition

            try:

                print('Val... file {0}'.format(filename))

                data = np.loadtxt(dataset + filename)

                print('Val... data size {}'.format(data.shape))

                x, y = process_dataset_file(data)

                print(x.shape)

                print(y.shape)

                X_val = np.vstack((X_val, x))

                y_val = np.concatenate([y_val, y])

            except KeyError:

                print('ERROR: Did not find {0} in zip file'.format(filename))

                

        else: # for test data sets

            # Testing partition

            try:

                print('Test... file {0}'.format(filename))

                data = np.loadtxt(dataset + filename)

                print('Test... data size {}'.format(data.shape))

                x, y = process_dataset_file(data)

                print(x.shape)

                print(y.shape)

                X_test = np.vstack((X_test, x))

                y_test = np.concatenate([y_test, y])

            except KeyError:

                print('ERROR: Did not find {0} in zip file'.format(filename))

            

        counter_files += 1 



    print("Final datasets with size: | train {0} | Val {1} | test {2} | ".format(X_train.shape,X_val.shape,X_test.shape))


    # np.savetxt("D:/Fourth Year/FYP/PAMAP2_Dataset/Processed_Data/test_target.dat", y_test)

    obj = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]

    f = open(os.path.join(target_filename), 'wb')
    
    cp.dump(obj, f, protocol=3)

#     cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)

    f.close()

    return X_train, y_train, X_val, y_val, X_test, y_test



def get_args():

    '''This function parses and return arguments passed in'''

    # Assign args to variables
    #Path to the extracted folder containing the pamap2 dataset
    # dataset = 'D:/Fourth Year/FYP/'
    # target_filename = 'D:/Fourth Year/FYP/PAMAP2_Dataset/Processed_Data/pamap23.data'

    # For GPU:
    dataset = '/data/mark/RawDatasets/'

    target_filename = '/data/mark/NetworkDatasets/baseline/pamap2.data'

    # Return all variable values
    return dataset, target_filename


if __name__ == '__main__':

    X_train = np.empty((0, NB_SENSOR_CHANNELS)) # 0 rows, 40 columns
    y_train = np.empty((0))
    
    X_val = np.empty((0, NB_SENSOR_CHANNELS))
    y_val = np.empty((0))
    
    X_test = np.empty((0, NB_SENSOR_CHANNELS))
    y_test = np.empty((0))

    pamap2_dataset, output = get_args()

    X_train, y_train, X_val, y_val, X_test, y_test = generate_data(pamap2_dataset, output)

    
    #     #get rid of +-6g acceleration columns as paper says not use them 

    # features_delete = np.arange(5, 8) 

    # features_delete = np.concatenate([features_delete, np.arange(18, 21)])

    # features_delete = np.concatenate([features_delete, np.arange(31, 34)])

    

    # X_train = np.delete(X_train, features_delete, 1) # removing along x axis (columns) this time
    # X_val = np.delete(X_val, features_delete, 1) # removing along x axis (columns) this time
    # X_test = np.delete(X_test, features_delete, 1) # removing along x axis (columns) this time
    
    # print("New: X_train shape {}".format(X_train.shape))
    # print("New: X_val shape {}".format(X_val.shape))
    # print("New: X_test shape {}".format(X_test.shape))
    
    T = 100
    s = 22
    
    # # to generate individual IMU plots
    # # ss.sliding_window(X_train, y_train, T, s, '/data/mark/NetworkDatasets/pamap2/Train/')
    # # ss.sliding_window(X_test, y_test, T, s, '/data/mark/NetworkDatasets/pamap2/Test/')
    # # ss.sliding_window(X_val, y_val, T, s, '/data/mark/NetworkDatasets/pamap2/Validation/')

    # # to generate individual IMU plots with heart rate sensor
    # ss.sliding_window_HR(X_train, y_train, T, s, '/data/mark/NetworkDatasets/pamap2_HR/Train/')
    # ss.sliding_window_HR(X_test, y_test, T, s, '/data/mark/NetworkDatasets/pamap2_HR/Test/')
    # ss.sliding_window_HR(X_val, y_val, T, s, '/data/mark/NetworkDatasets/pamap2_HR/Validation/')

    # # # to generate plots of IMUs stored together
    # # ss.cnn_sliding_window(X_train, y_train, T, s, '/data/mark/NetworkDatasets/pamap2_cnn/Train/')
    # # ss.cnn_sliding_window(X_test, y_test, T, s, '/data/mark/NetworkDatasets/pamap2_cnn/Test/')
    # # ss.cnn_sliding_window(X_val, y_val, T, s, '/data/mark/NetworkDatasets/pamap2_cnn/Validation/')


    # # # to generate plots of IMUs stored together with heart rate sensor
    # # ss.cnn_sliding_window_HR(X_train, y_train, T, s, '/data/mark/NetworkDatasets/pamap2_cnn_HR/Train/')
    # # ss.cnn_sliding_window_HR(X_test, y_test, T, s, '/data/mark/NetworkDatasets/pamap2_cnn_HR/Test/')
    # # ss.cnn_sliding_window_HR(X_val, y_val, T, s, '/data/mark/NetworkDatasets/pamap2_cnn_HR/Validation/')

    # to generate plots of IMUs stored together with heart rate sensor
    gp.sliding_window_mm(X_train, y_train, T, s, '/data/mark/NetworkDatasets/rgb_plots/Train/')
    gp.sliding_window_mm(X_val, y_val, T, s, '/data/mark/NetworkDatasets/rgb_plots/Validation/')
    gp.sliding_window_mm(X_test, y_test, T, s, '/data/mark/NetworkDatasets/rgb_plots/Test/')

    

    print('Done')