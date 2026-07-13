#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:49:27 2026

@author: randallclark
"""

import numpy as np
from pathlib import Path

try:
    from . import ECE_Classifier
    from . import PR_Classifier
    from . import Ensemble_Classifier
except ImportError:
    import ECE_Classifier
    import PR_Classifier
    import Ensemble_Classifier



class Hmode_Ml_Predictions():
    def __init__(self, Model_Choice, ShotRange, Times, Exclude_Non_Plasma = 1):
        """
        Parameters
        ----------
        Model_Choice : STRING
            Model Choice can be 'ECE', 'PR', or 'Ensemble'. Ensemble is the most restrictive requiring Accessible PR and ECE data,
            but is the most accurate. Accuracy of models trends as 'Ensemble' > 'PR' > 'ECE'. 'ECE' also has strict density limits.
        
        ShotRange : LIST of INTEGERS
            
        Times : LIST of INTEGERS (in ms)
            Standard recommendation is to set it from 100 ms to 10000 ms with time steps of 100 ms and set Exclude_Non_Plasma to 1
        
        Exclude_Non_Plasma : BOOLEAN
            Set to 1 to exclude labels after the plasma dies (ip < 200 kA), set to 0 to include all times (so long as data exists)
        """
        
        #Confirm that Model_Choice is accurate
        self.Models = ['ECE','PR','Ensemble']
        assert Model_Choice in self.Models, 'The Model Choice must be one of the following: \'ECE\', \'PR\', or \'Ensemble\''
        
        #Setup for Label Collection
        try:
            self.MODEL_DIR = Path(__file__).resolve().parent / "Models/"
        except NameError:
            self.MODEL_DIR = Path.cwd() / "Models/"
        self.Model_Choice = Model_Choice
        self.ShotRange = ShotRange
        self.Times = Times
        self.Times = np.asarray(Times)
        self.Exclude_Non_Plasma = Exclude_Non_Plasma
        
        
    def Predict(self):
        #ECE Model
        if 'ECE' == self.Model_Choice:
            ECE_Labels = ECE_Classifier.Collect_Labels(self.MODEL_DIR, self.ShotRange, self.Times, self.Exclude_Non_Plasma)
            self.Result = ECE_Labels
        
        #PR Model
        elif 'PR' == self.Model_Choice:
            PR_Labels = PR_Classifier.Collect_Labels(self.MODEL_DIR, self.ShotRange, self.Times, self.Exclude_Non_Plasma)
            self.Result = PR_Labels
            
        #Ensemble Model
        elif 'Ensemble' == self.Model_Choice:
            #ONNX_ECE = ort.InferenceSession(self.MODEL_DIR/"ECE_GradBoost_v1.onnx")
            #ONNX_PR = ort.InferenceSession(self.MODEL_DIR/"PR_GradBoost_v1.onnx")
            #ONNX_Kmeans_ECE = ort.InferenceSession(self.MODEL_DIR/"ECE_Kmeans_v1.onnx")
            #ONNX_Kmeans_PR = ort.InferenceSession(self.MODEL_DIR/"PR_Kmeans_v1.onnx")
            #Ave_ECE = np.load(self.MODEL_DIR/'Ave_Dist_ECE_v1.npy')
            #Cluster_ECE = np.load(self.MODEL_DIR/'Cluster_loc_ECE_v1.npy')
            #Ave_PR = np.load(self.MODEL_DIR/'Ave_Dist_PR_v1.npy')
            #Cluster_PR = np.load(self.MODEL_DIR/'Cluster_loc_PR_v1.npy')
            
            Ensemble_Labels = Ensemble_Classifier.Collect_Labels(self.MODEL_DIR, self.ShotRange, self.Times, self.Exclude_Non_Plasma)
            self.Result = Ensemble_Labels
        
        return self.Result
        
        
        
        
        
        
        
        
        
        