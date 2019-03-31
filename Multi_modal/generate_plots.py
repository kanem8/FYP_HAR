import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import pickle

def sliding_window_mm(X_data, y_data, T, s, target_filepath):

    # import matplotlib.gridspec as gridspec

    # with open('D:/Fourth Year/FYP/PAMAP2_Dataset/Processed_Data/pamap2.data', 'rb') as f:
    #         [(X_train, y_train), (X_val, y_val), (X_test, y_test)] = pickle.load(f)

    # X_data = X_train
    # y_data = y_train
    # T = 100
    # s = 22
    # target_filepath = 'D:/Fourth Year/FYP/PAMAP2_Dataset/Multi_modal/Train/'

    x_dataframe = pd.DataFrame(X_data)

    #fig, axes = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=True)

    #     rows = np.arange(0, range(X_data.shape[0])-100, s)
    x = X_data.shape[0] - T
    rows = list(range(0, x, s))
    # rows = list(range(0, 22, s))

    # plots = np.arange(2, 11, 1)
    # plots = np.arange(0, 4, 1)


    count_figures = 0

    figure_csv = target_filepath + "figure_labels.csv"
    file_csv = open(figure_csv, 'w')
    writer = csv.writer(file_csv)
    writer.writerow(['Fig_Number', 'Activity_Label'])

    print('About to begin plots')
    # j = 250000
    for j in rows:
        
        # if j > s*15:
        #     break

        label_idx = j

        # Don't plot figure of mixed labels
        if y_data[j] != y_data[j+T]:
            label_idx = j + int(T/2)
            #continue

        count_figures += 1

        # gs1 = gridspec.GridSpec(3, 1)
        # gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

        fig_IMU1, axes_IMU1 = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False)
        plt.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00, wspace=0, hspace=0)
        # plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=None, hspace=None)

        # plt.figure(figsize = (1.25,1.25))
        # fig_IMU1.set_size_inches(4,4)

        i = 2

        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[0], y=[i,i+1,i+2], c='r', legend=None)
        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[0], y=[i+3,i+4,i+5], c='g', legend=None)
        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[0], y=[i+6,i+7,i+8], c='b', legend=None)
        # x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[0], y=[i+6,i+7,i+8,i-1], c='b', legend=None)
        axes_IMU1[0].axis('off')

        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[1], y=[i+10,i+11,i+12], c='r', legend=None)
        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[1], y=[i+13,i+14,i+15], c='g', legend=None)
        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[1], y=[i+16,i+17,i+18], c='b', legend=None)
        # x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[1], y=[i+16,i+17,i+18,i-1+10], c='b', legend=None)
        axes_IMU1[1].axis('off')

        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[2], y=[i+20,i+21,i+22], c='r', legend=None)
        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[2], y=[i+23,i+24,i+25], c='g', legend=None)
        x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[2], y=[i+26,i+27,i+28], c='b', legend=None)
        # x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[2], y=[i+26,i+27,i+28,i+20-1], c='b', legend=None)
        axes_IMU1[2].axis('off')

        image_IMU1 = target_filepath + "fig" + str(count_figures) + ".jpg"
        fig_IMU1.savefig(image_IMU1)

        writer.writerow([('fig' + str(count_figures)), str(int(y_data[label_idx]))])

        plt.close('all')

    print('Finished Plots for ' + target_filepath)
    print('Total number of plots for this set of data: {0}'.format(str(count_figures))) 




def sliding_window_v2(pickle_file, T, s, target_filepath):

    with open(pickle_file, 'rb') as f:
            [(X_train, y_train), (X_val, y_val), (X_test, y_test)] = pickle.load(f)

    if target_filepath == '/data/mark/NetworkDatasets/vision_app2/Train/':
        X_data = X_train
        y_data = y_train

    elif target_filepath == '/data/mark/NetworkDatasets/vision_app2/Validation/':
        X_data = X_val
        y_data = y_val

    else:
        X_data = X_test
        y_data = y_test


    # T = 200
    # s = 50
    # target_filepath = 'D:/Fourth Year/FYP/PAMAP2_Dataset/Vision_app2/Train/'

    x_dataframe = pd.DataFrame(X_data)

    #fig, axes = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=True)

    #     rows = np.arange(0, range(X_data.shape[0])-100, s)
    x = X_data.shape[0] - T
    rows = list(range(0, x, s))
    # rows = list(range(0, 22, s))

    plots = np.arange(1, 14, 1)

    count_figures = 0

    figure_csv = target_filepath + "figure_labels.csv"
    file_csv = open(figure_csv, 'w')
    writer = csv.writer(file_csv)
    writer.writerow(['Fig_Number', 'Activity_Label'])

    print('About to begin plots')
    # j = 250000
    for j in rows:
        
        # if j > s*1:
        #     break

        label_idx = j

        # Don't plot figure of mixed labels
        if y_data[j] != y_data[j+T]:
            label_idx = j + int(T/2)
            #continue

        count_figures += 1

        # gs1 = gridspec.GridSpec(3, 1)
        # gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
        
        
        fig_IMU1, axes_IMU1 = plt.subplots(nrows=13, ncols=1, sharex=True, sharey=False)
        plt.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00, wspace=0, hspace=0)

        fig_IMU2, axes_IMU2 = plt.subplots(nrows=13, ncols=1, sharex=True, sharey=False)
        plt.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00, wspace=0, hspace=0)

        fig_IMU3, axes_IMU3 = plt.subplots(nrows=13, ncols=1, sharex=True, sharey=False)
        plt.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00, wspace=0, hspace=0)

            
        for i in plots:

            x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[i-1], y=i, legend=None)
            axes_IMU1[i-1].axis('off')

            x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU2[i-1], y=(i+13), legend=None)
            axes_IMU2[i-1].axis('off')

            x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU3[i-1], y=(i+26), legend=None)
            axes_IMU3[i-1].axis('off')

        image_IMU1 = target_filepath + "IMU_1Hand/fig" + str(count_figures) + ".jpg"
        fig_IMU1.savefig(image_IMU1)

        image_IMU2 = target_filepath + "IMU_2Chest/fig" + str(count_figures) + ".jpg"
        fig_IMU2.savefig(image_IMU2) 

        image_IMU3 = target_filepath + "IMU_3Ankle/fig" + str(count_figures) + ".jpg"
        fig_IMU3.savefig(image_IMU3)

        writer.writerow([('fig' + str(count_figures)), str(y_data[label_idx])])

        plt.close('all')

    print('Finished Plots')
    print('Number of plots per IMU: {0}'.format(str(count_figures)))
    print('Total number of plots for this set of data: {0}'.format(str(count_figures*3)))


if __name__ == '__main__':
    
    pickle_file = '/data/mark/NetworkDatasets/pamap2/pamap2.data'

    T = 200
    s = 50

    # to generate plots of IMUs stored together with heart rate sensor
    sliding_window_v2(pickle_file, T, s, '/data/mark/NetworkDatasets/vision_app2/Train/')
    sliding_window_v2(pickle_file, T, s, '/data/mark/NetworkDatasets/vision_app2/Validation/')
    sliding_window_v2(pickle_file, T, s, '/data/mark/NetworkDatasets/vision_app2/Test/')

    

    print('Done')