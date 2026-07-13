#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:51:13 2026

@author: randallclark
"""
import numpy as np
import onnxruntime as ort

try:
    from . import ECE_Classifier
    from . import PR_Classifier
except ImportError:
    import ECE_Classifier
    import PR_Classifier
    
ECE_CUTOFF = 3.5
PR_CUTOFF = 2.5

"""
Parameters
----------
MODEL_DIR - Path to Classifier Package
ShotRange - List of Shot Numbers to collect data from and label
Times - Times Labelling will occur so long as data is available
Exclude_Non_Plasma - Boolean parameter to exclude data after the plasma ends, if true

Returns
-------
A Dictionary of the Ensemble result, ECE result, and PR result. Since the Ensemble model requires the calculation of the other two models,
they are provided to the user but can be ignored. The Ensemble result is a dictionary whose keys are the shot numbers and contain nested dictionaries.
These nested dictionaries contain the times, shot numbers, Labels, Hmode probability, Lmode probability, and an Error string if there were errors.
"""
def Collect_Labels(MODEL_DIR,ShotRange, Times, Exclude_Non_Plasma):
    #The Ensemble Classifier will work by running the ECE and PR scripts to collect their label probabilities and then combine them
    #using the ensemble method of a weighted average that penalizes prediction probabilities for diverging from the training data behavior
    
    #Load in the ECE and PR Kmeans Models and associated Statistics
    ONNX_Kmeans_ECE = ort.InferenceSession(MODEL_DIR/"ECE_Kmeans_v1.onnx")
    ONNX_Kmeans_PR = ort.InferenceSession(MODEL_DIR/"PR_Kmeans_v1.onnx")
    Ave_ECE = np.load(MODEL_DIR/'Ave_Dist_ECE_v1.npy')
    Cluster_ECE = np.load(MODEL_DIR/'Cluster_loc_ECE_v1.npy')
    Ave_PR = np.load(MODEL_DIR/'Ave_Dist_PR_v1.npy')
    Cluster_PR = np.load(MODEL_DIR/'Cluster_loc_PR_v1.npy')
    
    
    #Collect the ECE and PR Model Predictions
    ECE_Results = ECE_Classifier.Collect_Labels(MODEL_DIR,ShotRange, Times, Exclude_Non_Plasma,retain_feature_vector = True)
    PR_Results = PR_Classifier.Collect_Labels(MODEL_DIR,ShotRange, Times, Exclude_Non_Plasma,retain_feature_vector = True)
    
    #We Must build a new Data Dictionary that combines the Data from ECE_Results and PR_Results at shots & times where both are valid
    DD = DataDictionaryBuilder(ECE_Results,PR_Results,ShotRange)
    
    #Evaluate the Esnemble model for all data in the Data Dictionary
    Ensemble_Result = BuildEnsembleDictionary(DD,ONNX_Kmeans_ECE,ONNX_Kmeans_PR,Ave_ECE,Cluster_ECE,Ave_PR,Cluster_PR)
    return {'Ensemble':Ensemble_Result, 'ECE':ECE_Results, 'PR':PR_Results}
    
"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
This Function builds Data Dictionary, evaluate matching times where both ECE and PR provide predcitions, and collects all this data into a Dictionary
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def DataDictionaryBuilder(ECE_Results,PR_Results,ShotRange):
    DD = {}
    for i, shot in enumerate(ShotRange):
        assert ECE_Results[i]['shot'] == PR_Results[i]['shot'], 'Shot Numbers should/must match -> developer bug, please report'
        
        shot = ECE_Results[i]['shot']
        Time = []
        ECE_Hmode_prob = []
        ECE_Feature_Vector = []
        PR_Hmode_prob = []
        PR_Feature_Vector = []
        
        common_times, ece_idx, pr_idx = np.intersect1d(
            ECE_Results[i]['Label_Times'],
            PR_Results[i]['Label_Times'],
            return_indices=True)
        
        for j, t in enumerate(common_times):
            ECE_idx = ece_idx[j]
            ECE_Hmode_prob.append(ECE_Results[i]['LH_Prob'][ECE_idx][1])
            ECE_Feature_Vector.append(ECE_Results[i]['Feature_Vector'][ECE_idx])
            
            PR_idx = pr_idx[j]
            PR_Hmode_prob.append(PR_Results[i]['LH_Prob'][PR_idx][1])
            PR_Feature_Vector.append(PR_Results[i]['Feature_Vector'][PR_idx])
            
            Time.append(t)
        
        Dict = {}
        Dict['shot'] = shot
        Dict['Time'] = np.asarray(Time)
        Dict['ECE_Hmode_prob'] = np.asarray(ECE_Hmode_prob)
        Dict['ECE_Feature_Vector'] = np.asarray(ECE_Feature_Vector)
        Dict['PR_Hmode_prob'] = np.asarray(PR_Hmode_prob)
        Dict['PR_Feature_Vector'] = np.asarray(PR_Feature_Vector)
        
        DD[shot] = Dict
    return DD
    
"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
This Function takes the results in the aforementioned Data Dictionary and calculates the ensemble model prediction using the available data
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def BuildEnsembleDictionary(DD,ONNX_Kmeans_ECE,ONNX_Kmeans_PR,Ave_ECE,Cluster_ECE,Ave_PR,Cluster_PR):
    Result = {}
    
    #ONNX Session Inputs and Outputs
    ECE_input_name = ONNX_Kmeans_ECE.get_inputs()[0].name
    ECE_output_names = [out.name for out in ONNX_Kmeans_ECE.get_outputs()]
    PR_input_name = ONNX_Kmeans_PR.get_inputs()[0].name
    PR_output_names = [out.name for out in ONNX_Kmeans_PR.get_outputs()]
    
    
    for key in DD.keys():
        #Create temporary Dictionary to build into the Result
        Dict = {}
        Error = ''
        
        Dict['shot'] = DD[key]['shot']
        Dict['Time'] = DD[key]['Time']
        try:
            if DD[key]['Time'].size == 0:
                Dict['Ensemble_Hmode_Prob'] = np.array([])
                Dict['Ensemble_Lmode_Prob'] = np.array([])
                Dict['Ensemble_Label'] = np.array([])
                Error = 'No overlapping valid ECE and PR times'
            else:
                #use the ONNX model to identify the index of the cluster the feature belongs to
                ECE_Features = DD[key]['ECE_Feature_Vector']
                ECEoutputs = ONNX_Kmeans_ECE.run(ECE_output_names, {ECE_input_name: ECE_Features.astype(np.float32)})
                PR_Features = DD[key]['PR_Feature_Vector']
                PRoutputs = ONNX_Kmeans_PR.run(PR_output_names, {PR_input_name: PR_Features.astype(np.float32)})
                
                #Calulate the L2 distance between that index cluster location and the feature itself
                ECE_Distances = np.linalg.norm(ECE_Features - Cluster_ECE[ECEoutputs[0]],axis = 1)
                PR_Distances = np.linalg.norm(PR_Features - Cluster_PR[PRoutputs[0]],axis = 1)
                
                #Put the probabilities, distnaces, average distance, and cuttoffs into the Ensemble Model for a probability output
                ECE_Prob = DD[key]['ECE_Hmode_prob']
                PR_Prob = DD[key]['PR_Hmode_prob']
                Ensemble_Hmode_Prob, Ensemble_Label = Ensemble_Model(ECE_Prob,PR_Prob,Ave_ECE,Ave_PR,ECE_Distances,PR_Distances)
                
                #Save the calculated Probability and Labels
                Dict['Ensemble_Hmode_Prob'] = Ensemble_Hmode_Prob
                Dict['Ensemble_Lmode_Prob'] = 1 - Ensemble_Hmode_Prob
                Dict['Ensemble_Label'] = Ensemble_Label
        except Exception as e:
            Error = f'Failed Ensemble: {e}'
        
        Dict['Error'] = Error
        Result[key] = Dict
    
    return Result

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
This Function performs the relatively simple mathematics of the ensemble now that the bulk of the hard work (model training, 
cluster evaluation, data preprocessing, and dictionary organizatio) is complete
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def Ensemble_Model(ECE_Prob,PR_Prob,Ave_ECE,Ave_PR,ECE_Distances,PR_Distances):
    Weight1 = np.minimum(1.0, (Ave_ECE * ECE_CUTOFF) / np.maximum(ECE_Distances, 1e-12))
    Weight2 = np.minimum(1.0, (Ave_PR * PR_CUTOFF) / np.maximum(PR_Distances, 1e-12))
    
    ProbTot = (ECE_Prob*Weight1 + PR_Prob*Weight2)/(Weight1+Weight2)
    return ProbTot, np.round(ProbTot)






