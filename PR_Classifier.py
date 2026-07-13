#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:51:13 2026

@author: randallclark
"""
from toksearch_d3d import PtDataSignal
from toksearch import Pipeline
from toksearch import MdsSignal
import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import BayesianRidge

"""
Parameters
----------
MODEL_DIR - Path to Classifier Package
ShotRange - List of Shot Numbers to collect data from and label
Times - Times Labelling will occur so long as data is available
Exclude_Non_Plasma - Boolean parameter to exclude data after the plasma ends, if true

Returns
-------
A Dictionary of each shot with their labels, time of labelling, H and L mode probabilities, shot number, and a dictionary of Errors detailing
why labels at requested timeslices during available data times didn't get labeled
"""
def Collect_Labels(MODEL_DIR,ShotRange,Times,Exclude_Non_Plasma,retain_feature_vector = False):
    #Collect ip and the Profile Reflectometer Densities, perform the precoessing of the data, collect densities from the polynomial spline fit,
    #And feed the spline fitted densities into the the pretrained ONNX gradient boosted Classifier Model
    
    ip_signal = PtDataSignal('ip')
    dims = ['times','radius']
    reflect_3drho_signal = MdsSignal(r'\reflect_3drho', 'electrons', dims = dims)
    
    pipeline = Pipeline(ShotRange)
    pipeline.fetch('ip', ip_signal)
    pipeline.fetch('reflect_3drho', reflect_3drho_signal)
    
    #Parametrization
    ip_minimum = 200000                                                         #in Ampers
    ip_begin_time = 500                                                         #in ms
    
    degree = 3                                                                  #Polynomial order of the Spline Fit
    n_knots = 10                                                                #Spline Fit parameter
    X = np.array([0,0.2,0.4,0.6,0.8,0.85,0.9,0.95,1.0,1.1]).reshape(-1, 1)      #Locations along Rho to collect features, Y, for classifier input
    
    @pipeline.map
    def CalcLabels(record):
        #We use the pipeline.map as part of the toksearch algorithm to do our labelling on the fly while data is collected and discarded for memory efficiency
        if record['reflect_3drho'] is None or record['ip'] is None:
            record['LH_Label'] = []
            record['LH_Prob'] = []
            record['Label_Times'] = []
            record['Labeling_Errors'] = 'None Data'
            record['Feature_Vector'] = []
        
        else:
            #Build a Data Dictionary for the shot at time Times with the ECE, BT, and Rmidout Data
            AvTimes = Available_Times(record,Times,Exclude_Non_Plasma,ip_minimum,ip_begin_time)
            DD, Error = Make_DataDictionary(record,AvTimes)
            
            #Collect the Weighted Coefficients for the RBFfit
            DD_Spline, Error = Spline_Fit_Calc(DD,Error,degree,n_knots,X)
            Spline_Values = np.array([v['Y'] for v in DD_Spline.values()])
            
            #Perform the Labelling using the ONNX Model
            ONNX_PR = ort.InferenceSession(MODEL_DIR/"PR_GradBoost_v1.onnx")
            input_name = ONNX_PR.get_inputs()[0].name
            output_names = [out.name for out in ONNX_PR.get_outputs()]
            
            if Spline_Values.size == 0:
                record['LH_Label'] = []
                record['LH_Prob'] = []
                record['Label_Times'] = []
            else:
                outputs = ONNX_PR.run(output_names, {input_name: Spline_Values.astype(np.float32)})
                record['LH_Label'] = outputs[0]
                record['LH_Prob'] = outputs[1]
                record['Label_Times'] = np.array([v['Time'] for v in DD_Spline.values()])
            
            record['Labeling_Errors'] = Error
            #Retain Feature Vector for Ensemble method calculation
            if retain_feature_vector:
                record['Feature_Vector'] = Spline_Values
    
    #To prevent bloat data from overloading a large labelling session, discard all data except for our labels and probabilities
    if retain_feature_vector:
        pipeline.keep(['LH_Label', 'LH_Prob','Label_Times','Labeling_Errors','shot','Feature_Vector'])
    else:
        pipeline.keep(['LH_Label', 'LH_Prob','Label_Times','Labeling_Errors','shot'])
    
    records = pipeline.compute_multiprocessing()
    
    return records


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
This Function evaluates which of the requested times are possible to collect (and have plasma if Exclude_Non_Plasma=1 / ip > 200kA)
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def Available_Times(record,RequestTimes,Exclude_Non_Plasma,ip_minimum,ip_begin_time):
    #Verify the minimum times and maximum times constrained by least available data, then check ip for plasma if Exclude_Non_Plasma=1
    Min_Time = np.min(record['reflect_3drho']['times'])
    
    if Exclude_Non_Plasma:
        #Check for when ip first drops below 200kA after the 500ms mark
        ip_times = record['ip']['times']
        ip_data = record['ip']['data']
        
        Five00_idx = np.where(ip_times>=ip_begin_time)[0][0]
        try:
            endtimeindex = np.where(abs(ip_data[Five00_idx:]) < ip_minimum)[0][0]
        except:
            endtimeindex = ip_data.shape[0]-1-Five00_idx
        ip_max_time = ip_times[endtimeindex+Five00_idx] 
    else:
        #Just let the ip_max_time be one of the other values if missing plasma isn't a concern, or if the method is hindering analysis
        ip_max_time = np.max(record['reflect_3drho']['times'])
    
    Max_Time = min(np.max(record['reflect_3drho']['times']),
                    ip_max_time)
    
    Times = RequestTimes[(RequestTimes >= Min_Time) & (RequestTimes <= Max_Time)]
    return Times

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
These functions build the Data Dictionary of PR data at the requested (and available) times
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def Make_DataDictionary(record,times):
    DD = {}
    Error = {}
    
    for t in times:
        try:
            Density, Rho = Collect_Density_Rho(t,record)
            Dict = {'Density':Density,
                    'Rho':Rho,
                    'Time':t}
            DD[t] = Dict
        
        except Exception as e:
            Error[t] = f'Bad PR Data at time {t}: {e}'
            
    return DD, Error

def Collect_Density_Rho(t,record):
    Density = record['reflect_3drho']['data']
    Rho = record['reflect_3drho']['radius']
    PR_time = record['reflect_3drho']['times']
    IDX = np.abs(PR_time - t).argmin()
    assert PR_time.shape[0] == Density.shape[1], 'Time Axis in Data must match length of Time Matrix'
    assert Rho.size == Density.size, 'Radius Matrix must match Density Matrix'
    return Density[:,IDX],Rho[:,IDX]

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
These functions data the Preprocessed Data Dictionary and perform the polynomial spline fits of the 1-D Density data for feature extraction
The features at the target points X, will be exported to the Binary Classifier model for prediction
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def Spline_Fit_Calc(DD,Error,degree,n_knots,X):
    Dict_XY = {}
    
    Time = np.array([v['Time'] for v in DD.values()])
    Density = [v['Density'] for v in DD.values()]
    Rho = [v['Rho'] for v in DD.values()]
    
    for i in range(Time.shape[0]):
        try:
            SplineModel,BayesianModel = Model_Maker(Rho[i],Density[i]/1e19,n_knots,degree)
            Y,Ystd = Predict_Y(X,SplineModel,BayesianModel)
            Dict_XY[i] = {'Time':Time[i],
                          'Y':Y}
        except Exception as e:
            Error[Time[i]] = f'Spline fit failed at time {Time[i]}: {e}'
    return Dict_XY, Error


def Model_Maker(X,y,n_knots,degree):
    X = X.reshape(-1, 1)

    # Spline Fit
    spline = SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", extrapolation="constant")
    Phi_spline = spline.fit_transform(X)

    # Fit Bayesian Ridge Regression
    Bayemodel = BayesianRidge()
    Bayemodel.fit(Phi_spline,y)
    return spline,Bayemodel
    
    
def Predict_Y(X,SplineModel,BayeModel):
    Phi_test = SplineModel.transform(X)
    y_mean, y_std = BayeModel.predict(Phi_test, return_std=True)
    return y_mean, y_std



