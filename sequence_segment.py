import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def sliding_window(X_data, y_data, T, s, target_filepath):
    
    """
    target_filepath is of form: eg. D:/Fourth Year/FYP/PAMAP2_Dataset/Segments/Train/
    
    """
    
    x_dataframe = pd.DataFrame(X_data)
    
    #fig, axes = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=True)

#     rows = np.arange(0, range(X_data.shape[0])-100, s)
    x = X_data.shape[0] - T
    rows = list(range(0, x, s))
    
    plots = np.arange(2, 11, 1)
    
    count_figures = 0
    
    figure_csv = target_filepath + "figure_labels.csv"
    file_csv = open(figure_csv, 'w')
    writer = csv.writer(file_csv)
    writer.writerow(['Fig_Number', 'Activity_Label'])
    
    print('About to begin plots')
    
    for j in rows:

        label_idx = j
        
        # Don't plot figure of mixed labels
        if y_data[j] != y_data[j+T]:
            label_idx = j + int(T/2)
            #continue
        
        count_figures += 1
        
        fig_IMU1, axes_IMU1 = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=False)
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=None, hspace=None)
        
        fig_IMU2, axes_IMU2 = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=False)
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=None, hspace=None)
        
        fig_IMU3, axes_IMU3 = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=False)
        plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=None, hspace=None)

        
        for i in plots:
            
            x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU1[i-2], y=i, legend=None)
            axes_IMU1[i-2].axis('off')

            x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU2[i-2], y=(i+10), legend=None)
            axes_IMU2[i-2].axis('off')

            x_dataframe.iloc[j:(j+T)].plot(ax=axes_IMU3[i-2], y=(i+20), legend=None)
            axes_IMU3[i-2].axis('off')

#             dataset.iloc[2929:3029].plot(ax=axes[1], y=[10,11,12], legend=None)
#             axes[1].axis('off')

#             dataset.iloc[2929:3029].plot(ax=axes[2], y=[13,14,15], legend=None)
#             axes[2].axis('off')

        #plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=None, hspace=None)
        image_IMU1 = target_filepath + "IMU_1Hand/fig" + str(count_figures) + ".jpg"
        fig_IMU1.savefig(image_IMU1)
        
        image_IMU2 = target_filepath + "IMU_2Chest/fig" + str(count_figures) + ".jpg"
        fig_IMU2.savefig(image_IMU2) 
        
        image_IMU3 = target_filepath + "IMU_3Ankle/fig" + str(count_figures) + ".jpg"
        fig_IMU3.savefig(image_IMU3)

        writer.writerow([('fig' + str(count_figures)), str(int(y_data[label_idx]))])
        
        plt.close('all')
    
    print('Finished Plots for ' + target_filepath)
    print('Number of plots per IMU: {0}'.format(str(count_figures)))
    print('Total number of plots for this set of data: {0}'.format(str(count_figures*3)))
