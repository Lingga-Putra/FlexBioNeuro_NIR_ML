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
from sklearn.model_selection import PredefinedSplit, GridSearchCV  # For cross-validation and hyperparameter tuning
from sklearn.neighbors import KNeighborsClassifier  # For k-Nearest Neighbors classification
from sklearn.naive_bayes import GaussianNB, BernoulliNB  # For Gaussian and Bernoulli Naive Bayes classification
from sklearn.decomposition import PCA, TruncatedSVD  # For dimensionality reduction
from sklearn.ensemble import ExtraTreesClassifier  # For Extra Trees classification
from sklearn.feature_selection import SelectKBest, f_classif  # For feature selection with chi-squared test
from sklearn.model_selection import StratifiedGroupKFold, cross_validate  # For cross-validation with group information
from sklearn.cross_decomposition import PLSRegression  # For Partial Least Squares regression
from scipy import signal  # For signal processing and detrending

# Machine learning classifiers
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # For Linear Discriminant Analysis
from sklearn.linear_model import LogisticRegression  # For Logistic Regression
from sklearn.svm import SVC  # For Support Vector Classification
from sklearn.ensemble import RandomForestClassifier  # For Random Forest classification
from sklearn.ensemble import GradientBoostingClassifier  # For Gradient Boosting classification
from sklearn.ensemble import AdaBoostClassifier  # For AdaBoost classification
from sklearn.neural_network import MLPClassifier  # For Multi-layer Perceptron classification


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


class DataProcessor:
    def __init__(self, data_file_path, calibration_file_path, meas_sec):
        """
        Initialize a DataProcessor object.

        Args:
            data_file_path (str): Path to the data file.
            calibration_file_path (str): Path to the calibration file.
            meas_sec (int): Measurement duration in seconds.
        """
        self.data_file_path        = data_file_path
        self.calibration_file_path = calibration_file_path
        self.meas_sec              = meas_sec
        self.df_all                = None
        self.X_all                 = None
        self.y_all                 = None
        self.groups                = None
        self.X_cal                 = None

    def load_data(self):
        """
        Load preprocessed variables from a CSV data file.
        """
        self.df_all = pd.read_csv(self.data_file_path, encoding='ISO-8859-1')

    def set_XY_groups(self):
        """
        Set X and Y values and groups based on the loaded data.
        """
        self.X_all  = self.df_all.iloc[:, :-1].values
        self.y_all  = self.df_all.iloc[:, -1].values
        self.groups = list(np.repeat(np.arange(1, int(np.shape(self.X_all)[0] / self.meas_sec) + 1), self.meas_sec))

    def load_calibration(self, calibrate="yes"):
        """
        Load calibration data and calculate absorbance (X_cal).
        """
        if calibrate == "yes":
            df_cal     = pd.read_csv(self.calibration_file_path, encoding='ISO-8859-1')
            cal_dark   = np.array(df_cal.iloc[0, :-1].values, dtype=np.float)
            cal_ref    = np.array(df_cal.iloc[1, :-1].values, dtype=np.float)
            self.X_cal = -np.log10((self.X_all - cal_dark) / (cal_ref - cal_dark))
        else:
            self.X_cal = self.X_all
        
    def randomize(self):
        """
        Randomly shuffle the data.
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
        X_features = np.hstack((X_S1_4, X_S1_7, X_S2_0, X_S2_2))
        return X_features
        



class Scaling_and_FeatureExtractor:
    def __init__(self, number_of_features, X_train_val_a, X_train_val_b, X_train_val_c, y_train_val, 
                 X_test_a, X_test_b, X_test_c, y_test, df_all):
        
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
    



class MLAlgorithm:
    def __init__(self, X_ml, y_ml, ps, n_jobs):
        """
        Initialize an MLAlgorithm object.

        Args:
            X_ml (numpy.ndarray): Input data for machine learning.
            y_ml (numpy.ndarray): Output data for machine learning.
            ps (PredefinedSplit): Predefined cross-validation split.
            n_jobs (int):         Number of CPU cores to use for parallel computation.
        """
        self.X_ml         = X_ml
        self.y_ml         = y_ml
        self.groups       = None
        self.ps           = ps
        self.n_jobs       = n_jobs
        self.grid_result  = None
        self.scores_train = None
        self.scores_test  = None

    def run_knn(self):
        """
        Run k-Nearest Neighbors (k-NN) algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_neighbors" : [2, 3, 4, 5, 6, 7],
            "weights"     : ["uniform", "distance"],
            "algorithm"   : ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size"   : [5, 10, 15],
            "p"           : [1, 2]
            }
        
        self.machine_learning_grid_search(KNeighborsClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "kNN")

    def run_gaussian_nb(self):
        """
        Run Gaussian Naive Bayes (GNB) algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "var_smoothing" : [1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            }

        self.machine_learning_grid_search(GaussianNB(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "GNB")

    def run_bernoulli_nb(self):
        """
        Run Bernoulli Naive Bayes algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "alpha"       : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "force_alpha" : [True, False],
            "binarize"    : [0.0, None],
            "fit_prior"   : [True, False]
            }

        self.machine_learning_grid_search(BernoulliNB(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "BernoulliNB")
        
    def run_decision_tree(self):
        """
        Run Decision Tree Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "criterion"         : ["gini", "entropy", "log_loss"],
            "splitter"          : ["best", "random"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
        	"min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
        	"max_features"      : [None, "sqrt", "log2"],
            "random_state"      : [42]
            }

        self.machine_learning_grid_search(DecisionTreeClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "DecisionTreeClassifier")
    
    def run_lda(self):
        """
        Run Linear Discriminant Analysis algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "solver"           : ["svd", "lsqr", "eigen"],
            "shrinkage"        : ["auto", None],
            "store_covariance" : [True, False],
        	"tol"              : [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
            }

        self.machine_learning_grid_search(LinearDiscriminantAnalysis(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "LinearDiscriminantAnalysis")
        
    def run_logreg(self):
        """
        Run Logistic Regression algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "penalty"    : ["l1", "l2", "elasticnet"],
            "dual"       : [True, False],
            "C"          : [0.01, 0.1, 1.0, 10.0],
        	"tol"        : [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5],
        	"solver"     : ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        	"max_iter"   : [5000],
        	"warm_start" : [True, False],
            }

        self.machine_learning_grid_search(LogisticRegression(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "LogisticRegression")
        
    def run_SVC(self):
        """
        Run Support Vector Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "kernel"       : ["linear", "poly", "rbf", "sigmoid"],
            "degree"       : [2, 3, 4],
            "C"            : [0.001, 0.01, 0.1, 1.0, 10.0],
        	"tol"          : [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4],
        	"gamma"        : ["scale", "auto"],
        	"probability"  : [True, False],
        	"shrinking"    : [True, False],
        	"class_weight" : ["balanced", None]
            }

        self.machine_learning_grid_search(SVC(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "SVC")
        
    def run_random_forest(self):
        """
        Run Random Forest Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_estimators"      : [3],
            "criterion"         : ["gini", "entropy", "log_loss"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
        	"min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
        	"max_features"      : [None, "sqrt", "log2"],
        	"bootstrap"         : [True, False],
        	"warm_start"        : [True, False],
        	"class_weight"      : [None],
            "random_state"      : [42]
            }
        
        self.machine_learning_grid_search(RandomForestClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "RandomForestClassifier")
        
    def run_extra_trees(self):
        """
        Run Extra Trees Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "n_estimators"      : [3],
            "criterion"         : ["gini", "entropy", "log_loss"],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
        	"min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
        	"max_features"      : [None, "sqrt", "log2"],
        	"bootstrap"         : [True, False],
        	"warm_start"        : [True, False],
        	"class_weight"      : [None],
            "random_state"      : [42]
            }
        
        self.machine_learning_grid_search(ExtraTreesClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "ExtraTreesClassifier")
        
    def run_grad_boosting(self):
        """
        Run Gradient Boosting Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "loss"              : ["log_loss", "exponential"],
            "learning_rate"     : [0.01, 0.1, 1.0],
            "n_estimators"      : [3],
            "criterion"         : ["friedman_mse", "squared_error"],
            "min_samples_split" : [2, 3, 4, 5, 6, 7, 8],
        	"min_samples_leaf"  : [1, 2, 3, 4, 5, 6, 7],
            "max_depth"         : [None, 5, 6, 7, 8, 9, 10],
            "max_features"      : [None, "sqrt", "log2"],
            "max_leaf_nodes"    : [None],
            "warm_start"        : [True, False],
        	"tol"               : [1e-3, 1e-4, 1e-5],
        	"n_iter_no_change"  : [None],
            "random_state"      : [42]
            }
        
        self.machine_learning_grid_search(GradientBoostingClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "GradientBoostingClassifier")
        
    def run_ada_boost(self):
        """
        Run Ada Boost Classifier algorithm with grid search for hyperparameter tuning.
        """
        parameters = {
            "learning_rate" : [0.01, 0.1, 1.0, 10.0],
            "n_estimators"  : [150],
        	"algorithm"     : ["SAMME", "SAMME.R"],
            "random_state"  : [42]
            }
        
        self.machine_learning_grid_search(AdaBoostClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "AdaBoostClassifier")
        
    def run_MLP(self):
        """
        Run Multiple Layer Perceptor Classifier algorithm with grid search for hyperparameter tuning.
        """ 
        parameters = {
            "hidden_layer_sizes" : [(10,), (13,), (16,)],
            "activation"         : ["identity", "logistic", "tanh", "relu"],
            "solver"             : ["lbfgs", "sgd", "adam"],
            "learning_rate"      : ["constant", "invscaling", "adaptive"],
            "random_state"       : [42],
            "max_iter"           : [5000],
            "alpha"              : [0.0001,0.01,0.1],
            }
        
        self.machine_learning_grid_search(MLPClassifier(), parameters)
        self.machine_learning_best_scores(self.grid_result.best_estimator_, "MLPClassifier")
        

    def machine_learning_grid_search(self, model, parameters):
        """
        Perform grid search for hyperparameter tuning.

        Args:
            model:             A machine learning model.
            parameters (dict): Hyperparameters and their possible values.
        """
        # Search Grid
        grid = GridSearchCV(estimator  = model,
                            param_grid = parameters,
                            cv         = self.ps,
                            n_jobs     = self.n_jobs,
                            scoring    = 'balanced_accuracy',
                            refit      = True)
        self.grid_result = grid.fit(self.X_ml, self.y_ml, groups=self.groups)

        # Summarize results
        print("Best: %f using %s" % (self.grid_result.best_score_, self.grid_result.best_params_))

    def machine_learning_best_scores(self, mod_best_param, model_name):
        """
        Calculate and print best scores for the selected machine learning model.

        Args:
            mod_best_param:   The best machine learning model obtained from hyperparameter tuning.
            model_name (str): The name of the machine learning model.
        """
        scores = cross_validate(mod_best_param, self.X_ml, self.y_ml, cv=self.ps, groups=self.groups,
                                scoring=('accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'),
                                return_train_score=True)

        self.scores_train = np.mean(scores['train_balanced_accuracy'])
        self.scores_test  = np.mean(scores['test_balanced_accuracy'])

        print(model_name)
        print(f'Score on training set:   balanced_accuracy={self.scores_train * 100:.1f}%')
        print(f'Score on validation set: balanced_accuracy={self.scores_test * 100:.1f}%')
        print(' ')



#%% Main
if __name__ == "__main__":
    """
    ######################################################################################################
    # Calibration case: VFA/TA-ratio or acetic acid concentration
    ######################################################################################################
    """
    cal_case = "Ac_acid"
    if cal_case == "VFA_TA":
        data_file_path = "Data/classification_NIR_Data_raw_VFA_TA.csv"
        calibrate      = "no"
    elif cal_case == "Ac_acid":
        data_file_path = "Data/classification_NIR_Data_raw_Ac_acid.csv"
        calibrate      = "yes"
    
    
    
    
    
    """
    ######################################################################################################
    # Processing data, Class: DataProcessor
    ######################################################################################################
    """
    # Initialize Data Processor
    data_processor = DataProcessor(data_file_path        = data_file_path,
                                   calibration_file_path = "Data/calibration_NIR_Data.csv",
                                   meas_sec              = 8)

    # Load raw data, set up features and targets, load calibration data, and perform baseline correction
    data_processor.load_data()
    data_processor.set_XY_groups()
    data_processor.load_calibration(calibrate = calibrate)
    data_processor.randomize()
    X_features_a = data_processor.baseline_correction(method=2)
    X_features_b = data_processor.baseline_correction(method=3)
    X_features_c = data_processor.baseline_correction(method=4)
    
    
    
    
    
    """
    ######################################################################################################
    # FOR loop preparation
    ######################################################################################################
    """
    # Initialize Stratified Group K-Fold for cross-validation
    gkf = StratifiedGroupKFold(n_splits=5)
    
    # Initialize lists to store hyperparameters and mean scores for training and validation sets
    choose_hyperparam = []
    mean_scores_train = list()
    mean_scores_test  = list()
    
    # Iterate through cross-validation folds
    for train_ix, test_ix in gkf.split(data_processor.X_cal,data_processor.y_all,data_processor.groups):
        # Split data for the current fold
        X_train_val_a, X_test_a = X_features_a[train_ix, :],      X_features_a[test_ix, :]
        X_train_val_b, X_test_b = X_features_b[train_ix, :],      X_features_b[test_ix, :]
        X_train_val_c, X_test_c = X_features_c[train_ix, :],      X_features_c[test_ix, :]
        y_train_val,   y_test   = data_processor.y_all[train_ix], data_processor.y_all[test_ix]
        
        
        
        
        
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
                                                          y_train_val        = y_train_val, 
                                                          X_test_a           = X_test_a, 
                                                          X_test_b           = X_test_b, 
                                                          X_test_c           = X_test_c, 
                                                          y_test             = y_test, 
                                                          df_all             = data_processor.df_all)
        
        features_extractor.Scaling_data()
        features_extractor.extract_features()
        features_extractor.Predefined_cv()
        
        
        
        
        
        """
        ######################################################################################################
        # Run Machine Learning algrotihm, Class: MLAlgorithm
        ######################################################################################################
        """
        # Machine Learning Algorithm Setup
        ml_algorithm = MLAlgorithm(X_ml   = features_extractor.X_ml, 
                                   y_ml   = features_extractor.y_ml, 
                                   ps     = features_extractor.ps, 
                                   n_jobs = 6)
    
        # Initialize the dictionary to map algorithm names to functions
        algorithm_mapping = {
            "knn":           ml_algorithm.run_knn,
            "gaussian_nb":   ml_algorithm.run_gaussian_nb,
            "bernoulli_nb":  ml_algorithm.run_bernoulli_nb,
            "decision_tree": ml_algorithm.run_decision_tree,
            "lda":           ml_algorithm.run_lda,
            "logreg":        ml_algorithm.run_logreg,
            "SVC":           ml_algorithm.run_SVC,
            "random_forest": ml_algorithm.run_random_forest,
            "extra_trees":   ml_algorithm.run_extra_trees,
            "grad_boosting": ml_algorithm.run_grad_boosting,
            "ada_boost":     ml_algorithm.run_ada_boost,
            "MLP":           ml_algorithm.run_MLP
        }
        
        # Specify the chosen algorithm (e.g., "knn", "gaussian_nb", "bernoulli_nb")
        chosen_algorithm = "decision_tree"  # Change this to select the desired algorithm
    
        # Check if the chosen algorithm exists in the mapping, then call it
        if chosen_algorithm in algorithm_mapping:
            algorithm_mapping[chosen_algorithm]()
        else:
            print(f"Algorithm '{chosen_algorithm}' is not recognized.")
        
        
        
        
        
        """
        ######################################################################################################
        # Save results
        ######################################################################################################
        """
        # Store the results, hyperparameters, and mean scores for this fold
        if not len(choose_hyperparam):
            name_hyperparam   = list(ml_algorithm.grid_result.best_params_.keys())
            choose_hyperparam = list(ml_algorithm.grid_result.best_params_.values())
            if chosen_algorithm == "MLP":
                choose_hyperparam[2] = str(choose_hyperparam[2]) # change tuple to string, otherwise does not work
        else:
            choose_hyperparam2 = list(ml_algorithm.grid_result.best_params_.values())
            if chosen_algorithm == "MLP":
                choose_hyperparam2[2] = str(choose_hyperparam2[2]) # change tuple to string, otherwise does not work
            choose_hyperparam = np.column_stack((choose_hyperparam, choose_hyperparam2))
            
        mean_scores_train.append(ml_algorithm.scores_train)
        mean_scores_test.append(ml_algorithm.scores_test)
        
        # Delete variables to save memory
        del X_train_val_a, X_train_val_b, X_train_val_c, y_train_val, 
        del X_test_a, X_test_b, X_test_c, y_test, train_ix, test_ix
        
    # Print results
    print("")
    print("")
    print("Calibration for :  " + cal_case)
    print("Chosen Algorithm:  " + chosen_algorithm)
    print('mean_scores_train, Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_train)*100, np.std(mean_scores_train)*100))
    print('mean_scores_test,  Balanced_Accuracy: %.2f (%.2f)' % (np.mean(mean_scores_test)*100, np.std(mean_scores_test)*100))
    
    print("")
    for i in range(len(name_hyperparam)):
        print("Hyperparameter " + name_hyperparam[i] + " = " + str(most_common(choose_hyperparam[i])))

    
    
    
    