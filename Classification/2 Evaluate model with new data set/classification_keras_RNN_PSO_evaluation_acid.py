# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:45:51 2023

@author: ge23hum
"""

# Data manipulation and analysis
import pandas as pd  # For data handling and analysis
import numpy as np  # For numerical operations
import copy  # For creating deep copies of data

# Data visualization
import matplotlib.pyplot as plt  # For creating plots and visualizations
import matplotlib.patches as mpatches  # For creating patches in plots
import matplotlib as mpl  # For customizing Matplotlib behavior

# Data preprocessing and machine learning
from sklearn.preprocessing import MinMaxScaler  # For feature scaling
from sklearn.model_selection import PredefinedSplit  # For cross-validation and hyperparameter tuning
from sklearn.decomposition import PCA, TruncatedSVD  # For dimensionality reduction
from sklearn.ensemble import ExtraTreesClassifier  # For Extra Trees classification
from sklearn.feature_selection import SelectKBest, f_classif  # For feature selection with chi-squared test
from sklearn.model_selection import cross_validate  # For cross-validation with group information
from sklearn.cross_decomposition import PLSRegression  # For Partial Least Squares regression
from scipy import signal  # For signal processing and detrending

# Machine learning classifiers
from keras.models import Sequential                 # forward neural network
from keras.layers import Dense                      # fully convoluteed
from keras import initializers
from keras.layers import LSTM                       # recurrent neural network


# Machine learning metrics evaluation
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score


# Machine learning evaluation
from sklearn.metrics import confusion_matrix            # confusion matrix
import seaborn as sns
from sklearn.model_selection import learning_curve      # learning curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



# Choose hyperparameter from most common elements
from statistics import mode
def most_common(List):
	return(mode(List))

# To ignore all warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

# to create the same random sequence every time
np.random.seed(42)

import random
random.seed(42) 

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.utils import set_random_seed
set_random_seed(42)



class SpectralPlotter:
    def __init__(self, show_plot="no"):
        """
        Initialize a SpectralPlotter object.

        Args:
            show_plot (str): A string indicating whether to display the plot. Should be 'yes' or 'no'.
        """
        self.show_plot = show_plot
        self.S1_4_WL   = range(1100,1352,2)
        self.S1_7_WL   = range(1350,1652,2)
        self.S2_0_WL   = range(1550,1952,2)
        self.S2_2_WL   = range(1750,2152,2)
    
    def plot_based_on_classes(self, all_WL, all_I, class_all, plot_label, title, xlabel, ylabel):
        """
        Plot data based on different classes.

        Args:
            all_WL (array-like):    Array of wavelength values.
            all_I (array-like):     Array of intensity values.
            class_all (array-like): Array of class labels for each data point.
            plot_label (list):      List of labels for the plot legend.
            title (str):            Title for the plot.
            xlabel (str):           Label for the x-axis.
            ylabel (str):           Label for the y-axis.

        Returns:
            fig (matplotlib.figure.Figure or None): The generated figure object if show_plot is 'yes', otherwise None.
        """
        
        # Edit the font and axes width
        mpl.rcParams['font.family']    = 'serif' # 'Avenir','serif'
        plt.rcParams['axes.linewidth'] = 1

        # Edit the font size and set the color of both classes
        font_size_title  = 6 + 2
        font_size_ylabel = 6 + 2
        font_size_xlabel = 6 + 2
        font_size_ticks  = 4 + 2
        font_size_legend = 4 + 2
        grid_line_width  = 0.5
        plot_line_width  = 0.5
        plot_marker      = 'None'
        plot_ls_0        = 'solid'
        plot_ls_1        = 'dashed'
        plot_marker_sz   = 1.0
        alpha_value      = 0.5

        pop_a = mpatches.Patch(color='green', label=plot_label[0])
        pop_b = mpatches.Patch(color='magenta', label=plot_label[1])
        
        # show_plot = 'yes' or 'no'
        if self.show_plot == "yes":
            # Create figure object and store it in a variable called 'fig'
            fig = plt.figure(figsize=(7.00, 3.50), dpi=300)

            for i in range(np.shape(all_I)[0]):
                if class_all[i] == 0:
                    plt.plot(all_WL,
                             all_I[i],
                             color='green',
                             linewidth=plot_line_width,
                             marker=plot_marker,
                             linestyle=plot_ls_0,
                             markersize=plot_marker_sz,
                             alpha=alpha_value)
                elif class_all[i] == 1:
                    plt.plot(all_WL,
                             all_I[i],
                             color='magenta',
                             linewidth=plot_line_width,
                             marker=plot_marker,
                             linestyle=plot_ls_1,
                             markersize=plot_marker_sz,
                             alpha=alpha_value)
                    
            # Set x-label, y-label and title
            plt.ylabel(ylabel, fontsize=font_size_ylabel)
            plt.xlabel(xlabel, fontsize=font_size_xlabel)
            plt.title(title, fontsize=font_size_title)
            
            # Set legend
            plt.legend(handles=[pop_a, pop_b], prop={'size': font_size_legend}, loc='upper right')

            # Hide the top and right spines of the axis
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            
            # Edit the major ticks of the x and y axes
            plt.gca().xaxis.set_tick_params(which='major', direction='in', top=False)
            plt.gca().yaxis.set_tick_params(which='major', direction='in', right=False)
            plt.tick_params(axis='both', which='major', labelsize=font_size_ticks)
            
            # Set the grid
            plt.gca().yaxis.grid(True, linewidth=grid_line_width, ls='dotted')
            
            # Set xticks
            plt.xticks([])

            end_S1_4_WL = len(self.S1_4_WL)
            end_S1_7_WL = len(self.S1_4_WL) + len(self.S1_7_WL)
            end_S2_0_WL = len(self.S1_4_WL) + len(self.S1_7_WL) + len(self.S2_0_WL)
            end_S2_2_WL = len(self.S1_4_WL) + len(self.S1_7_WL) + len(self.S2_0_WL) + len(self.S2_2_WL)

            x_ticks_pos = [0, int(len(self.S1_4_WL) / 2), end_S1_4_WL - 1,
                           end_S1_4_WL, end_S1_4_WL + int(len(self.S1_7_WL) / 2), end_S1_7_WL - 1,
                           end_S1_7_WL, end_S1_7_WL + int(len(self.S2_0_WL) / 2), end_S2_0_WL - 1,
                           end_S2_0_WL, end_S2_0_WL + int(len(self.S2_2_WL) / 2), end_S2_2_WL - 1]

            x_ticks_label = [str(self.S1_4_WL[0]), "...", str(self.S1_4_WL[-1]),
                             str(self.S1_7_WL[0]), "...", str(self.S1_7_WL[-1]),
                             str(self.S2_0_WL[0]), "...", str(self.S2_0_WL[-1]),
                             str(self.S2_2_WL[0]), "...", str(self.S2_2_WL[-1])]

            plt.xticks(x_ticks_pos, x_ticks_label)
            ticklabels = plt.gca().get_xticklabels()
            
            # set the x-ticks alignment
            ticklabels[0].set_ha("left")
            ticklabels[2].set_ha("right")
            ticklabels[3].set_ha("left")
            ticklabels[5].set_ha("right")
            ticklabels[6].set_ha("left")
            ticklabels[8].set_ha("right")
            ticklabels[9].set_ha("left")
            ticklabels[-1].set_ha("right")
            
            # show line, which separates the sensors
            y_min, y_max = plt.gca().get_ylim()
            plt.vlines(x=end_S1_4_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            plt.vlines(x=end_S1_7_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            plt.vlines(x=end_S2_0_WL, ymin=y_min, ymax=y_max, colors='blue', zorder=10)
            
            # vertical shading to separate the sensors better
            plt.axvspan(end_S1_4_WL, end_S1_7_WL - 1, alpha=0.1)
            plt.axvspan(end_S2_0_WL, end_S2_2_WL - 1, alpha=0.1)

            fig.set_tight_layout(True)
            plt.show()
        else:
            fig = None
        return fig






class DataProcessor:
    def __init__(self, data_file_path, test_data_file_path, calibration_file_path, meas_sec):
        """
        Initialize a DataProcessor object.

        Args:
            data_file_path (str): Path to the data file.
            calibration_file_path (str): Path to the calibration file.
            meas_sec (int): Measurement duration in seconds.
        """
        self.data_file_path        = data_file_path
        self.test_data_file_path   = test_data_file_path
        self.calibration_file_path = calibration_file_path
        self.meas_sec              = meas_sec
        self.df_all                = None
        self.X_all                 = None
        self.y_all                 = None
        self.X_cal                 = None
        self.X_features            = None
        self.df_test               = None
        self.X_test                = None
        self.y_test                = None

    def load_data(self):
        """
        Load preprocessed variables from a CSV data file.
        """
        self.df_all = pd.read_csv(self.data_file_path, encoding='ISO-8859-1')

    def load_test_data(self):
        """
        Load preprocessed variables from a CSV data file.
        """
        self.df_test = pd.read_csv(self.test_data_file_path, encoding='ISO-8859-1')

    def set_XY_values(self):
        """
        Set X and Y values based on the loaded data.
        """
        self.X_all  = self.df_all.iloc[:, :-1].values
        self.y_all  = self.df_all.iloc[:, -1].values
        
        self.X_test = self.df_test.iloc[:, :-1].values
        self.y_test = self.df_test.iloc[:, -1].values

    def load_calibration(self, calibrate="yes"):
        """
        Load calibration data and calculate absorbance (X_cal).
        """
        if calibrate == "yes":
            df_cal     = pd.read_csv(self.calibration_file_path, encoding='ISO-8859-1')
            cal_dark   = np.array(df_cal.iloc[0, :-1].values, dtype=np.float)
            cal_ref    = np.array(df_cal.iloc[1, :-1].values, dtype=np.float)
            self.X_cal  = -np.log10((self.X_all - cal_dark) / (cal_ref - cal_dark))
            self.X_test = -np.log10((self.X_test - cal_dark) / (cal_ref - cal_dark))
        else:
            self.X_cal = self.X_all
        
    def randomize(self):
        """
        Randomly shuffle the training data.
        """
        # Generate a random permutation of indices for data shuffling
        permutation = np.random.permutation(int(self.y_all.shape[0] / self.meas_sec))
        
        # Create a permutation group to apply to both X and Y
        permutation_group = np.repeat(permutation * 0, self.meas_sec)
        
        # Populate the permutation group with corresponding indices
        for i in range(len(permutation)):
            permutation_group[i * self.meas_sec:(i + 1) * self.meas_sec] = np.arange(permutation[i] * self.meas_sec, (permutation[i] + 1) * self.meas_sec)
        
        # Shuffle the data based on the permutation
        self.X_cal = self.X_cal[permutation_group, :]
        self.y_all = self.y_all[permutation_group]
        
        del permutation, permutation_group
        
        """
        Randomly shuffle the test data.
        """
        # Generate a random permutation of indices for data shuffling
        permutation = np.random.permutation(int(self.y_test.shape[0] / self.meas_sec))
        
        # Create a permutation group to apply to both X and Y
        permutation_group = np.repeat(permutation * 0, self.meas_sec)
        
        # Populate the permutation group with corresponding indices
        for i in range(len(permutation)):
            permutation_group[i * self.meas_sec:(i + 1) * self.meas_sec] = np.arange(permutation[i] * self.meas_sec, (permutation[i] + 1) * self.meas_sec)
        
        # Shuffle the data based on the permutation
        self.X_test = self.X_test[permutation_group, :]
        self.y_test = self.y_test[permutation_group]
        
        del permutation, permutation_group
        
    def msc(self, input_data, reference=None):
        ''' Perform Multiplicative scatter correction
        
        Args:
            input_data (numpy.ndarray): Input data to be corrected.
            reference (numpy.ndarray, optional): Reference spectrum. If not provided, it's estimated from the mean.
        
        Returns:
            tuple: A tuple containing the corrected data and the reference spectrum.
        '''
     
        # Mean center correction
        for i in range(input_data.shape[0]):
            input_data[i,:] -= input_data[i,:].mean()
     
        # Get the reference spectrum. If not given, estimate it from the mean    
        if reference is None:    
            ref = np.mean(input_data, axis=0)
        else:
            ref = reference
     
        # Define a new array and populate it with the corrected data    
        data_msc = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            # Run regression
            fit = np.polyfit(ref, input_data[i,:], 1, full=True)
            # Apply correction
            data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
     
        return (data_msc, ref)

    def snv(self, input_data):
        ''' Perform Standard Normal Variate (SNV) correction
        
        Args:
            input_data (numpy.ndarray): Input data to be corrected.
        
        Returns:
            numpy.ndarray: Corrected data using SNV.
        '''
        # Define a new array and populate it with the corrected data  
        output_data = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            # Apply correction
            output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
     
        return output_data
    
    def split_blocks(self, X):
        ''' Split input data into blocks for different sensors
        
        Args:
            X (numpy.ndarray): Input data to be split.
        
        Returns:
            tuple: A tuple containing separate data blocks for each sensor.
        '''
        # Length of each sensor's data
        len_S1_4 = len(range(1100, 1352, 2))
        len_S1_7 = len(range(1350, 1652, 2))
        len_S2_0 = len(range(1550, 1952, 2))
        len_S2_2 = len(range(1750, 2152, 2))
        
        end_S1_4 = len_S1_4
        end_S1_7 = len_S1_4 + len_S1_7
        end_S2_0 = len_S1_4 + len_S1_7 + len_S2_0
        end_S2_2 = len_S1_4 + len_S1_7 + len_S2_0 + len_S2_2
        
        # Split data for each sensor
        X_S1_4   = X[:, 0:end_S1_4]
        X_S1_7   = X[:, end_S1_4:end_S1_7]
        X_S2_0   = X[:, end_S1_7:end_S2_0]
        X_S2_2   = X[:, end_S2_0:end_S2_2]
        
        return X_S1_4, X_S1_7, X_S2_0, X_S2_2
    
    def baseline_correction(self, method):
        """
        Apply baseline correction to training data.

        Args:
            method (int): The method to use for baseline correction.
        """
        X_S1_4, X_S1_7, X_S2_0, X_S2_2 = self.split_blocks(copy.deepcopy(self.X_cal))
        
        if method == 1:
            # Detrending data
            X_S1_4 = signal.detrend(X_S1_4, axis=1)
            X_S1_7 = signal.detrend(X_S1_7, axis=1)
            X_S2_0 = signal.detrend(X_S2_0, axis=1)
            X_S2_2 = signal.detrend(X_S2_2, axis=1)
        elif method == 2:
            # Subtract mean from the data
            X_S1_4 = X_S1_4 - np.tile(np.mean(X_S1_4, axis=1), (X_S1_4.shape[1], 1)).T
            X_S1_7 = X_S1_7 - np.tile(np.mean(X_S1_7, axis=1), (X_S1_7.shape[1], 1)).T
            X_S2_0 = X_S2_0 - np.tile(np.mean(X_S2_0, axis=1), (X_S2_0.shape[1], 1)).T
            X_S2_2 = X_S2_2 - np.tile(np.mean(X_S2_2, axis=1), (X_S2_2.shape[1], 1)).T
        elif method == 3:
            # SNV
            for i in range(int(self.X_cal.shape[0] / self.meas_sec)):
                X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :])
        elif method == 4:
            # MSC
            for i in range(int(self.X_cal.shape[0] / self.meas_sec)):
                (X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :])
        
        # Combine the data again
        X_train_baseline_correction = np.hstack((X_S1_4, X_S1_7, X_S2_0, X_S2_2))
        
        """
        Apply baseline correction to test data.

        Args:
            method (int): The method to use for baseline correction.
        """
        X_S1_4, X_S1_7, X_S2_0, X_S2_2 = self.split_blocks(copy.deepcopy(self.X_test))
        
        if method == 1:
            # Detrending data
            X_S1_4 = signal.detrend(X_S1_4, axis=1)
            X_S1_7 = signal.detrend(X_S1_7, axis=1)
            X_S2_0 = signal.detrend(X_S2_0, axis=1)
            X_S2_2 = signal.detrend(X_S2_2, axis=1)
        elif method == 2:
            # Subtract mean from the data
            X_S1_4 = X_S1_4 - np.tile(np.mean(X_S1_4, axis=1), (X_S1_4.shape[1], 1)).T
            X_S1_7 = X_S1_7 - np.tile(np.mean(X_S1_7, axis=1), (X_S1_7.shape[1], 1)).T
            X_S2_0 = X_S2_0 - np.tile(np.mean(X_S2_0, axis=1), (X_S2_0.shape[1], 1)).T
            X_S2_2 = X_S2_2 - np.tile(np.mean(X_S2_2, axis=1), (X_S2_2.shape[1], 1)).T
        elif method == 3:
            # SNV
            for i in range(int(self.X_cal.shape[0] / self.meas_sec)):
                X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :])
                X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :] = self.snv(X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :])
        elif method == 4:
            # MSC
            for i in range(int(self.X_cal.shape[0] / self.meas_sec)):
                (X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S1_4[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S1_7[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S2_0[self.meas_sec*i:self.meas_sec*(i+1), :])
                (X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :], _) = self.msc(X_S2_2[self.meas_sec*i:self.meas_sec*(i+1), :])
        
        # Combine the data again
        X_test_baseline_correction = np.hstack((X_S1_4, X_S1_7, X_S2_0, X_S2_2))
        
        return X_train_baseline_correction, X_test_baseline_correction
        



class Scaling_and_FeatureExtractor:
    def __init__(self, number_of_features, X_train_val_a, X_train_val_b, X_train_val_c, y_train_val, 
                 X_test_a, X_test_b, X_test_c, y_test, df_all, meas_sec):
        
        self.number_of_features = number_of_features
        self.X_train_val_a      = X_train_val_a
        self.X_train_val_b      = X_train_val_b
        self.X_train_val_c      = X_train_val_c
        self.y_train_val        = y_train_val
        self.X_test_a           = X_test_a
        self.X_test_b           = X_test_b
        self.X_test_c           = X_test_c
        self.y_test             = y_test
        self.df_all             = df_all
        self.meas_sec           = meas_sec
        self.X_out              = None
        self.X_out_test         = None
        self.X_ml               = None
        self.y_ml               = None
        self.ps                 = None
        
    def Scaling_data(self):
        """
        Scale input data using MinMaxScaler.
        """
        scaler_train_val_a = MinMaxScaler(feature_range=(0, 1))
        self.X_train_val_a = scaler_train_val_a.fit_transform(self.X_train_val_a)
        self.X_test_a      = scaler_train_val_a.transform(self.X_test_a)
        
        scaler_train_val_b = MinMaxScaler(feature_range=(0, 1))
        self.X_train_val_b = scaler_train_val_b.fit_transform(self.X_train_val_b)
        self.X_test_b      = scaler_train_val_b.transform(self.X_test_b)
        
        scaler_train_val_c = MinMaxScaler(feature_range=(0, 1))
        self.X_train_val_c = scaler_train_val_c.fit_transform(self.X_train_val_c)
        self.X_test_c      = scaler_train_val_c.transform(self.X_test_c)
        

    def extra_trees_feature_reduction(self, X_train_val, y_train_val, df_all, X_test):
        """
        Perform feature reduction using Extra Trees Classifier.

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            df_all (pd.DataFrame):       Dataframe containing all features.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data and the selected feature names.
        """
        # Feature reduction using Extra Trees Classifier
        model                   = ExtraTreesClassifier()
        model.fit(X_train_val, y_train_val)
        feat_importances        = pd.Series(model.feature_importances_, index=df_all.columns[:-1])

        # take few best wavelengths (dfcolumns_out) and 
        # save these measurements (X_out) from these best wavelengths
        dfcolumns_out           = []
        best_feature_importance = feat_importances.nlargest(self.number_of_features)
        ind_best_feature        = np.argwhere(df_all.columns[:-1]==best_feature_importance.index[0])[0][0]
        X_out_FR                = copy.deepcopy(X_train_val[:,ind_best_feature])
        X_out_test_FR           = copy.deepcopy(X_test[:,ind_best_feature])
        dfcolumns_out.append(df_all.columns[ind_best_feature])
        for i in range(self.number_of_features-1):
            ind_best_feature = np.argwhere(df_all.columns[:-1]==best_feature_importance.index[i+1])[0][0]
            X_out_FR         = np.column_stack((X_out_FR, X_train_val[:,ind_best_feature]))
            X_out_test_FR    = np.column_stack((X_out_test_FR, X_test[:,ind_best_feature]))
            dfcolumns_out.append(df_all.columns[ind_best_feature])

        return X_out_FR, X_out_test_FR, dfcolumns_out

    def univariate_selection(self, X_train_val, y_train_val, df_all, X_test):
        """
        Perform feature selection using the ANOVA F-test.

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            df_all (pd.DataFrame):       Dataframe containing all features.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data and the selected feature names.
        """
        # Univariate feature selection using chi-squared test
        bestfeatures              = SelectKBest(score_func=f_classif, k=self.number_of_features)
        fit                       = bestfeatures.fit(X_train_val, y_train_val)
        dfcolumns                 = pd.DataFrame(df_all.columns[:-1])
        dfscores                  = pd.DataFrame(fit.scores_)

        # Concatenate two dataframes for better visualization
        featureScores             = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns     = ['Specs', 'Score']

        # take 10 best wavelengths (dfcolumns_out) and 
        # save these measurements (X_out) from these best wavelengths
        dfcolumns_out           = []
        best_univariate_selection = featureScores.nlargest(self.number_of_features,'Score')
        # -----------------------------------------------------------
        ind_best_feature        = best_univariate_selection.index[0]
        X_out_US                = copy.deepcopy(X_train_val[:,ind_best_feature])
        X_out_test_US           = copy.deepcopy(X_test[:,ind_best_feature])
        dfcolumns_out.append(df_all.columns[ind_best_feature])
        for i in range(1,self.number_of_features):
            ind_best_feature = best_univariate_selection.index[i]
            X_out_US         = np.column_stack((X_out_US, X_train_val[:,ind_best_feature]))
            X_out_test_US    = np.column_stack((X_out_test_US, X_test[:,ind_best_feature]))
            dfcolumns_out.append(df_all.columns[ind_best_feature])

        return X_out_US, X_out_test_US, dfcolumns_out

    def pls(self, X_train_val, y_train_val, X_test):
        """
        Perform feature reduction using Partial Least Squares (PLS).

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            y_train_val (numpy.ndarray): Training and validation output data.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data.
        """
        # Partial Least Squares (PLS) feature reduction
        plsda          = PLSRegression(n_components=self.number_of_features)
        (X_out_PLS,_)  = plsda.fit_transform(X_train_val, y_train_val)
        X_out_test_PLS = plsda.transform(X_test)

        return X_out_PLS, X_out_test_PLS

    def pca(self, X_train_val, X_test):
        """
        Perform feature reduction using Principal Component Analysis (PCA).

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data.
        """
        # Principal Component Analysis (PCA) feature reduction
        sklearn_pca    = PCA(n_components=self.number_of_features)
        X_out_PCA      = sklearn_pca.fit_transform(X_train_val)
        X_out_test_PCA = sklearn_pca.transform(X_test)

        return X_out_PCA, X_out_test_PCA

    def truncated_svd(self, X_train_val, X_test):
        """
        Perform feature reduction using Truncated Singular Value Decomposition (TruncatedSVD).

        Args:
            X_train_val (numpy.ndarray): Training and validation input data.
            X_test (numpy.ndarray):      Test input data.

        Returns:
            tuple: A tuple containing reduced training and test data.
        """
        # Truncated Singular Value Decomposition (TruncatedSVD) feature reduction
        sklearn_svd    = TruncatedSVD(n_components=self.number_of_features)
        X_out_SVD      = sklearn_svd.fit_transform(X_train_val)
        X_out_test_SVD = sklearn_svd.transform(X_test)

        return X_out_SVD, X_out_test_SVD

    def extract_features(self):
        """
        Extract and reduce features using various methods. (a)
        """
        X_out_FR,  X_out_test_FR, _ = self.extra_trees_feature_reduction(self.X_train_val_a, self.y_train_val, self.df_all, self.X_test_a)
        X_out_US,  X_out_test_US, _ = self.univariate_selection(self.X_train_val_a, self.y_train_val, self.df_all, self.X_test_a)
        X_out_PLS, X_out_test_PLS   = self.pls(self.X_train_val_a, self.y_train_val, self.X_test_a)
        X_out_PCA, X_out_test_PCA   = self.pca(self.X_train_val_a, self.X_test_a)
        X_out_SVD, X_out_test_SVD   = self.truncated_svd(self.X_train_val_a, self.X_test_a)

        # Combine the selected features from different methods
        X_out_a      = np.hstack((X_out_FR,      X_out_US,      X_out_PLS,      X_out_PCA,      X_out_SVD))
        X_out_test_a = np.hstack((X_out_test_FR, X_out_test_US, X_out_test_PLS, X_out_test_PCA, X_out_test_SVD))
        
        del X_out_FR,  X_out_test_FR, X_out_US, X_out_test_US, X_out_PLS, X_out_test_PLS
        del X_out_PCA, X_out_test_PCA, X_out_SVD, X_out_test_SVD
        
        """
        Extract and reduce features using various methods. (b)
        """
        X_out_FR,  X_out_test_FR, _ = self.extra_trees_feature_reduction(self.X_train_val_b, self.y_train_val, self.df_all, self.X_test_b)
        X_out_US,  X_out_test_US, _ = self.univariate_selection(self.X_train_val_b, self.y_train_val, self.df_all, self.X_test_b)
        X_out_PLS, X_out_test_PLS   = self.pls(self.X_train_val_b, self.y_train_val, self.X_test_b)
        X_out_PCA, X_out_test_PCA   = self.pca(self.X_train_val_b, self.X_test_b)
        X_out_SVD, X_out_test_SVD   = self.truncated_svd(self.X_train_val_b, self.X_test_b)

        # Combine the selected features from different methods
        X_out_b      = np.hstack((X_out_FR,      X_out_US,      X_out_PLS,      X_out_PCA,      X_out_SVD))
        X_out_test_b = np.hstack((X_out_test_FR, X_out_test_US, X_out_test_PLS, X_out_test_PCA, X_out_test_SVD))
        
        del X_out_FR,  X_out_test_FR, X_out_US, X_out_test_US, X_out_PLS, X_out_test_PLS
        del X_out_PCA, X_out_test_PCA, X_out_SVD, X_out_test_SVD
        
        """
        Extract and reduce features using various methods. (c)
        """
        X_out_FR,  X_out_test_FR, _ = self.extra_trees_feature_reduction(self.X_train_val_c, self.y_train_val, self.df_all, self.X_test_c)
        X_out_US,  X_out_test_US, _ = self.univariate_selection(self.X_train_val_c, self.y_train_val, self.df_all, self.X_test_c)
        X_out_PLS, X_out_test_PLS   = self.pls(self.X_train_val_c, self.y_train_val, self.X_test_c)
        X_out_PCA, X_out_test_PCA   = self.pca(self.X_train_val_c, self.X_test_c)
        X_out_SVD, X_out_test_SVD   = self.truncated_svd(self.X_train_val_c, self.X_test_c)

        # Combine the selected features from different methods
        X_out_c      = np.hstack((X_out_FR,      X_out_US,      X_out_PLS,      X_out_PCA,      X_out_SVD))
        X_out_test_c = np.hstack((X_out_test_FR, X_out_test_US, X_out_test_PLS, X_out_test_PCA, X_out_test_SVD))
        
        del X_out_FR,  X_out_test_FR, X_out_US, X_out_test_US, X_out_PLS, X_out_test_PLS
        del X_out_PCA, X_out_test_PCA, X_out_SVD, X_out_test_SVD
        
        self.X_out      = np.hstack((X_out_a,      X_out_b,      X_out_c))
        self.X_out_test = np.hstack((X_out_test_a, X_out_test_b, X_out_test_c))
        
        
        
    def Predefined_cv(self):
        """
        Perform predefined cross-validation.
        """
        # The indices which have the value -1 will be kept in train.
        train_indices = np.full((np.size(self.y_train_val),), -1, dtype=int)
    
        # The indices which have zero or positive values, will be kept in test
        val_indices   = np.full((np.size(self.y_test),), 0, dtype=int)
        test_fold     = np.append(train_indices, val_indices)
    
        # Predefined
        self.ps       = PredefinedSplit(test_fold)
        self.X_ml     = np.concatenate((self.X_out,       self.X_out_test))
        self.y_ml     = np.concatenate((self.y_train_val, self.y_test))
        
        
    def reshape_data(self):
        """
        Reshape input data arrays to match the desired format for recurrent neural networks.
    
        This function reshapes the input features and target labels to create sequences suitable for training
        recurrent neural networks. It's especially useful for time series data where the input features are
        organized into sequences with a fixed length (e.g., time steps).
    
        Returns:
            None
        """
        self.X_out       = self.X_out.reshape(int(self.X_out.shape[0]/self.meas_sec), self.meas_sec, self.X_out.shape[1])
        self.y_train_val = self.y_train_val.reshape(len(self.y_train_val),1)[::self.meas_sec]
    
        self.X_out_test  = self.X_out_test.reshape(int(self.X_out_test.shape[0]/self.meas_sec), self.meas_sec, self.X_out_test.shape[1])
        self.y_test      = self.y_test.reshape(len(self.y_test),1)[::self.meas_sec]
    


class NeuralNetworkOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, meas_sec, cal_case, epochs=1000, show_plot="no"):
        """
        Initialize the NeuralNetworkOptimizer class.

        Args:
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.
            meas_sec (int): Measurement section value.
            epochs (int): Number of training epochs for neural networks (default is 1000).
            show_plot (str): Whether to display plots ("yes" or "no").

        Attributes:
            meas_sec (int): Measurement section value.
            epochs (int): Number of training epochs for neural networks.
            show_plot (str): Whether to display plots ("yes" or "no").
            X_train (numpy.ndarray): Training input features.
            y_train (numpy.ndarray): Training target labels.
            X_test (numpy.ndarray): Test input features.
            y_test (numpy.ndarray): Test target labels.
        """
        self.meas_sec  = meas_sec
        self.epochs    = epochs
        self.show_plot = show_plot
        self.cal_case  = cal_case
        self.X_train   = X_train
        self.y_train   = y_train
        self.X_test    = X_test
        self.y_test    = y_test
        
        
    def create_baseline(self):
        """
        Create a baseline deep neural network model.

        Returns:
            keras.models.Sequential: Compiled Keras model for the deep neural network.
        """
        if self.cal_case == "VFA_TA":
            # generate model
            model = Sequential()
            model.add(LSTM(units                 = 7, 
                           input_shape           = (self.meas_sec, self.X_train.shape[2]), 
                           activation            = "softplus",
                           recurrent_activation  = "softsign",
                           kernel_initializer    = initializers.GlorotUniform(seed=42),
                           recurrent_initializer = initializers.GlorotUniform(seed=42),
                           bias_initializer      = initializers.VarianceScaling(seed=42),
                           kernel_regularizer    = "L2",
                           bias_regularizer      = "L2",
                           activity_regularizer  = "L1L2",
                           kernel_constraint     = "UnitNorm",
                           bias_constraint       = "UnitNorm",
                           return_sequences      = False))
        
            model.add(Dense(units                = 1,
                            activation           = "sigmoid", 
                            kernel_initializer   = initializers.GlorotUniform(seed=42),
                            bias_initializer     = initializers.VarianceScaling(seed=42),
                            kernel_regularizer   = "L2",
                            bias_regularizer     = "L2",
                            activity_regularizer = "L1L2",
                            kernel_constraint    = "UnitNorm",
                            bias_constraint      = "UnitNorm"))
            
            # compile model
            model.compile(loss='binary_crossentropy', optimizer="Nadam", metrics=['binary_accuracy'])
            
        elif self.cal_case == "Ac_acid":
            # generate model
            model = Sequential()
            model.add(LSTM(units                 = 9, 
                           input_shape           = (self.meas_sec, self.X_train.shape[2]), 
                           activation            = "softsign",
                           recurrent_activation  = "tanh",
                           kernel_initializer    = initializers.LecunUniform(seed=42),
                           recurrent_initializer = initializers.LecunUniform(seed=42),
                           bias_initializer      = initializers.GlorotNormal(seed=42),
                           kernel_regularizer    = "L1L2",
                           bias_regularizer      = "L1L2",
                           activity_regularizer  = "L1L2",
                           kernel_constraint     = "NonNeg",
                           bias_constraint       = "NonNeg",
                           return_sequences      = False))
        
            model.add(Dense(units                = 1,
                            activation           = "sigmoid", 
                            kernel_initializer   = initializers.LecunUniform(seed=42),
                            bias_initializer     = initializers.GlorotNormal(seed=42),
                            kernel_regularizer   = "L1L2",
                            bias_regularizer     = "L1L2",
                            activity_regularizer = "L1L2",
                            kernel_constraint    = "NonNeg",
                            bias_constraint      = "NonNeg"))
            
            # compile model
            model.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['binary_accuracy'])
        
        return model
    
    def evaluate_nn(self):
        """
        Train and evaluate the deep neural network model.

        This method trains the deep neural network, calculates evaluation scores, and displays plots.

        Returns:
            None
        """
        estimator = self.create_baseline()
        estimator.fit(x               = self.X_train, 
                      y               = self.y_train, 
                      epochs          = self.epochs, 
                      batch_size      = 32,
                      verbose         = 0,
                      validation_data = (self.X_test, self.y_test))
        
        self.machine_learning_best_scores(estimator, "Deep Neural Network")
        self.plot_confusion_matrix(estimator, "Deep Neural Network")
        self.plot_learning_curve(self.create_baseline(), "Deep Neural Network")
        self.plot_ROC_curves_and_AUC(estimator, "Deep Neural Network")
        
        
        
    def machine_learning_best_scores(self, estimator, model_name):
        """
        Calculate and display machine learning evaluation scores.

        Args:
            estimator (keras.models.Sequential): Trained deep neural network model.
            model_name (str): Name of the machine learning model.

        Returns:
            None
        """
        preds_train = (estimator.predict(self.X_train) > 0.5).astype("int32")
        preds_test  = (estimator.predict(self.X_test) > 0.5).astype("int32")
            
        train1      = balanced_accuracy_score(self.y_train[:,0],preds_train[:,0])
        test1       = balanced_accuracy_score(self.y_test[:,0],preds_test[:,0])
        
        train2      = precision_score(self.y_train[:,0],preds_train[:,0])
        test2       = precision_score(self.y_test[:,0],preds_test[:,0])
    
        train3      = recall_score(self.y_train[:,0],preds_train[:,0])
        test3       = recall_score(self.y_test[:,0],preds_test[:,0])
        
        train4      = f1_score(self.y_train[:,0],preds_train[:,0])
        test4       = f1_score(self.y_test[:,0],preds_test[:,0])
        
        data      = [[f'{train1*100:.2f}%', f'{train2*100:.2f}%', f'{train3*100:.2f}%', f'{train4*100:.2f}%'],
                    [f'{test1*100:.2f}%', f'{test2*100:.2f}%', f'{test3*100:.2f}%', f'{test4*100:.2f}%']]
        index     = ["training-validation set", "test set"]
        columns   = ["balanced_accuracy", "precision", "recall", "f1-score"]
        df_scores = pd.DataFrame(data, index, columns)
        print(model_name)
        print(df_scores)
        print(' ')

        
    
    def plot_learning_curve(self, estimator, model_name):
        """
        Plot the learning curve for the training and validation sets.

        Args:
            estimator (keras.models.Sequential): Trained deep neural network model.
            model_name (str): Name of the machine learning model.

        Returns:
            None
        """
        # Check if the plot should be displayed
        if self.show_plot == "yes":
            fontsize_title  = 10
            fontsize_xlabel = 8
            fontsize_ylabel = 8
            fontsize_xtick  = 6
            fontsize_ytick  = 6
            fontsize_legend = 6
            grid_line_width = 0.5
            
            plt.rc('font', family = 'serif')
            plt.rc('xtick', labelsize = fontsize_xtick)
            plt.rc('ytick', labelsize = fontsize_ytick)
            plt.rcParams['figure.dpi'] = 450
            plt.rcParams["figure.autolayout"] = True
            
            history = estimator.fit(x               = self.X_train, 
                                    y               = self.y_train, 
                                    epochs          = self.epochs, 
                                    batch_size      = 32,
                                    verbose         = 0,
                                    validation_data = (self.X_test, self.y_test))
        
    
            history_dict    = history.history
            loss_values     = history_dict['loss']
            val_loss_values = history_dict['val_loss']
            accuracy        = history_dict['binary_accuracy']
            val_accuracy    = history_dict['val_binary_accuracy']
            epochs          = range(1, len(loss_values) + 1)
            
            fig = plt.figure(figsize=(4.00,3.00))
            ax = fig.subplots()
            
            # Hide the top and right spines of the axis
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Edit the major and minor ticks of the x and y axes
            ax.xaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', top=False)
            ax.yaxis.set_tick_params(which='major', size=5, width=1.0, direction='in', right=False)
    
            ax.set_title(model_name, fontsize = fontsize_title)
            
            ax.set_xlabel("Epochs", fontsize = fontsize_xlabel)
            ax.set_ylabel("Score", fontsize = fontsize_ylabel)
            
            # Plot learning curve
            ax.grid(True, linewidth = grid_line_width, ls = '--')
            ax.plot(
                epochs, accuracy, color="magenta", label="Training accuracy")
            ax.plot(
                epochs, val_accuracy, color="green", label="Validation accuracy")
            ax.legend(loc="lower right", fontsize = fontsize_legend)
            
            # Set ylim
            ax.set_ylim(0, 1)
            
            
            plt.tight_layout()
            plt.show()
    
    
    def plot_confusion_matrix(self, mod_best_param, model_name):
        """
        Plot a confusion matrix for a binary classification model and visualize the predicted versus actual outcomes.
    
        This function generates a confusion matrix and visualizes it as a heatmap to assess the performance of a binary classification model.
    
        Args:
            mod_best_param: The trained binary classification model.
            model_name: A string representing the name of the model.
    
        Returns:
            None
        """
        y_pred_test = (mod_best_param.predict(self.X_test) > 0.5).astype("int32")
        cf_matrix   = confusion_matrix(self.y_test[:,0], y_pred_test[:,0])
        
        if self.show_plot == "yes":
            plt.figure(figsize=(4.00,3.00))
            
            #cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
            cmn = cf_matrix.astype('int')
            
            #ax = sns.heatmap(cmn*100, annot=True, fmt='.1f', vmin=0, vmax=100, cmap="Blues")
            ax = sns.heatmap(cmn, annot=True, vmin=0, vmax=cf_matrix.max(), cmap="YlGnBu")
            
            ax.set_title(model_name);
            ax.set_xlabel('\nPredicted Values')
            ax.set_ylabel('Actual Values ');
            
            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(['Class 0','Class 1'])
            ax.yaxis.set_ticklabels(['Class 0','Class 1'])
            
            ## Display the visualization of the Confusion Matrix.
            plt.tight_layout()
            plt.show()
            
    def plot_ROC_curves_and_AUC(self, mod_best_param, model_name):
        """
        Plots ROC curves and calculates the AUC (Area Under the Curve) scores for a binary classification model.
    
        Args:
            mod_best_param (object): The best-performing binary classification model.
            model_name (str): The name of the model for labeling the ROC curve plot.
    
        Returns:
            None
    
        This function generates a "no skill" prediction (majority class) as a baseline and compares it with the provided model's predictions.
        It calculates the ROC AUC scores for both models and, if specified, displays a ROC curve plot.
        """
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(self.y_test))]
        
        # predict probabilities
        lr_probs = mod_best_param.predict(self.X_test)
        
        # calculate scores
        ns_auc = roc_auc_score(self.y_test, ns_probs)
        lr_auc = roc_auc_score(self.y_test, lr_probs[:,0])
        
        # summarize scores
        print('Baseline: ROC AUC=%.3f' % (ns_auc))
        print('ML algorithm: ROC AUC=%.3f' % (lr_auc))
        
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, lr_probs)
        
        if self.show_plot == "yes":
            plt.figure(figsize=(4.00,3.00))
        
            # plot the roc curve for the model
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
            
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(model_name)
            
            # show the legend
            plt.legend()
            
            # show the plot
            plt.show()





#%% Main
if __name__ == "__main__":
    """
    ######################################################################################################
    # Calibration case: VFA/TA-ratio or acetic acid concentration
    ######################################################################################################
    """
    cal_case = "VFA_TA"
    if cal_case == "VFA_TA":
        data_file_path      = "Data/classification_NIR_Data_raw_VFA_TA.csv"
        test_data_file_path = "Data/Test_classification_NIR_Data_raw_VFA_TA.csv"
        calibrate           = "no"
    elif cal_case == "Ac_acid":
        data_file_path      = "Data/classification_NIR_Data_raw_Ac_acid.csv"
        test_data_file_path = "Data/Test_classification_NIR_Data_raw_Ac_acid.csv"
        calibrate           = "yes"
    
    
    
    
    
    """
    ######################################################################################################
    # Processing data, Class: DataProcessor
    ######################################################################################################
    """
    # Initialize Data Processor
    data_processor = DataProcessor(data_file_path        = data_file_path,
                                   test_data_file_path   = test_data_file_path,
                                   calibration_file_path = "Data/calibration_NIR_Data.csv",
                                   meas_sec              = 8)

    # Load raw data, set up features and targets, load calibration data, and perform baseline correction
    data_processor.load_data()
    data_processor.load_test_data()
    data_processor.set_XY_values()
    data_processor.load_calibration(calibrate = calibrate)
    data_processor.randomize()
    X_train_val_a, X_test_a = data_processor.baseline_correction(method=2)
    X_train_val_b, X_test_b = data_processor.baseline_correction(method=3)
    X_train_val_c, X_test_c = data_processor.baseline_correction(method=4)
    
        
        
        
        
        
    """
    ######################################################################################################
    # Extract features, Class: Scaling_and_FeatureExtractor
    ######################################################################################################
    """
    # Feature Extraction and Predefined Cross-Validation Setup
    features_extractor = Scaling_and_FeatureExtractor(number_of_features = 5, 
                                                      X_train_val_a      = X_train_val_a, 
                                                      X_train_val_b      = X_train_val_b, 
                                                      X_train_val_c      = X_train_val_c, 
                                                      y_train_val        = data_processor.y_all, 
                                                      X_test_a           = X_test_a, 
                                                      X_test_b           = X_test_b, 
                                                      X_test_c           = X_test_c, 
                                                      y_test             = data_processor.y_test, 
                                                      df_all             = data_processor.df_all,
                                                      meas_sec           = data_processor.meas_sec)
        
    features_extractor.Scaling_data()
    features_extractor.extract_features()
    features_extractor.reshape_data()
        
        
        
        
        
    """
    ######################################################################################################
    # Evaluate Machine Learning, Class: NeuralNetworkOptimizer
    ######################################################################################################
    """
    # Machine Learning Algorithm Setup
    nn_optimizer = NeuralNetworkOptimizer(X_train   = features_extractor.X_out, 
                                          y_train   = features_extractor.y_train_val, 
                                          X_test    = features_extractor.X_out_test, 
                                          y_test    = features_extractor.y_test, 
                                          meas_sec  = data_processor.meas_sec,
                                          epochs    = 500,
                                          show_plot = "yes",
                                          cal_case  = cal_case)
    
    # Evaluate the chosen neural network
    nn_optimizer.evaluate_nn()

    
    
    
    