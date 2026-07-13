#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:51:12 2026

@author: randallclark
"""
from toksearch_d3d import PtDataSignal
from toksearch import Pipeline
from toksearch import MdsSignal
import numpy as np
import onnxruntime as ort

ECE_CHANNELS = [f"tece{i:02d}" for i in range(1, 40)] + ["tece40"]

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
def Collect_Labels(MODEL_DIR,ShotRange, Times, Exclude_Non_Plasma,retain_feature_vector = False):
    #To Collect the ECE Labels, we must collect the ECE Data from the FDP, evaluate the times ECE data is available and follows the
    #ECE Density rules, pre process the available data, send it through the ONNX classifier, and record the results while disregarding extra data
    
    
    #Collect ip, density, BT, Rmidout, and the 40 ECE Signals
    ip_signal = PtDataSignal('ip')
    Density_signal = MdsSignal(r'\Density','efit01')
    BT_signal = PtDataSignal('BT')
    Rmidout_signal = MdsSignal(r'\Rmidout','efit01')
    
    pipeline = Pipeline(ShotRange)
    pipeline.fetch('ip', ip_signal)
    pipeline.fetch('DENSITY', Density_signal)
    pipeline.fetch('BT', BT_signal)
    pipeline.fetch('Rmidout', Rmidout_signal)
    
    for ece in ECE_CHANNELS:
        pipeline.fetch(ece, MdsSignal(rf'\{ece}','ece'))
    
    #Parameterization
    ip_minimum = 200000             #in Ampers
    ip_begin_time = 500             #in ms
    
    Density_cutoff_value = 4.5e13   #in 1/cm^3
    remove_high_density = True      #Boolean
    remove_Rmax=True                #Boolean
    DeltaR = 0.1                    #Unit of Normalized Rho
    
    RBFparam = 1e2                  #Dimensionless Radial Basis Function Parameter
    CenterNum = 7                   #Number of RBF Centers to Fit
    include_00 = True               #Boolean
    maxtemp_loss = 0.90             #Percentage of keV
    
    @pipeline.map
    def CalcLabels(record):
        #We use the pipeline.map as part of the toksearch algorithm to do our labelling on the fly while data is collected and discarded for memory efficiency
        if (record['ip'] is None
        or record['DENSITY'] is None
        or record['BT'] is None
        or record['Rmidout'] is None
        or any(record[ece] is None for ece in ECE_CHANNELS)):
            record['LH_Label'] = []
            record['LH_Prob'] = []
            record['Label_Times'] = []
            record['Labeling_Errors'] = 'None Data'
            record['Feature_Vector'] = []
        
        else:
            #Build a Data Dictionary for the shot at time Times with the ECE, BT, and Rmidout Data
            AvTimes = Available_Times(record,Times,Exclude_Non_Plasma,ip_minimum,ip_begin_time)
            DD, Error = Make_DataDictionary(record,AvTimes,Density_cutoff_value,remove_high_density,remove_Rmax,DeltaR)
            
            #Collect the Weighted Coefficients for the RBFfit
            DD_Coef, Error = calc_all_coefficients(DD,Error,RBFparam,CenterNum,include_00,maxtemp_loss)
            
            #Perform the Labelling using the ONNX Model
            ONNX_ECE = ort.InferenceSession(MODEL_DIR/"ECE_GradBoost_v1.onnx")
            input_name = ONNX_ECE.get_inputs()[0].name
            output_names = [out.name for out in ONNX_ECE.get_outputs()]
            
            Coef = np.array([v['Weights'] for v in DD_Coef.values()])
            if Coef.size == 0:
                record['LH_Label'] = []
                record['LH_Prob'] = []
                record['Label_Times'] = []
            else:
                outputs = ONNX_ECE.run(output_names, {input_name: Coef.astype(np.float32)})
                record['LH_Label'] = outputs[0]
                record['LH_Prob'] = outputs[1]
                record['Label_Times'] = np.array([v['Time'] for v in DD_Coef.values()])
            
            record['Labeling_Errors'] = Error
            #Retain Feature Vector for Ensemble method calculation
            if retain_feature_vector:
                record['Feature_Vector'] = Coef
    
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
    Min_Time = max(np.min(record['DENSITY']['times']),
                    np.min(record['tece01']['times']),
                    np.min(record['BT']['times']),
                    np.min(record['Rmidout']['times']))
    
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
        ip_max_time = np.max(record['DENSITY']['times'])
    
    Max_Time = min(np.max(record['DENSITY']['times']),
                    np.max(record['tece01']['times']),
                    np.max(record['BT']['times']),
                    np.max(record['Rmidout']['times']),
                    ip_max_time)
    Times = RequestTimes[(RequestTimes >= Min_Time) & (RequestTimes <= Max_Time)]
    
    return Times

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
This Function builds Data Dictionary so that data can be verified as acceptable and collected at desired times to be delivered to the ML tools
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def Make_DataDictionary(record,Times,Density_cutoff_value,remove_high_density,remove_Rmax,DeltaR):
    DD = {}
    Error = {}
    
    for t in Times:
        try:
            Data = Collect_ECE(t,record)
            Density = Collect_Density(t,record)
            BT = Collect_BT(t,record)
            Rmidout = Collect_Rmidout(t,record)
            
            Dict = {'Data':Data,
                    'Time':t,
                    'Density':Density,
                    'BT':BT,
                    'Rmidout':Rmidout}
            DD[t] = Dict
        except Exception as e:
            Error[t] = f'Bad Data at time {t}: {e}'

    if remove_high_density:
        BadData = []
        for key in DD.keys():
            if DD[key]['Density'] > Density_cutoff_value:
                BadData.append(key)
                Error[key] = 'High Density at time '+str(DD[key]['Time'])
        for key in BadData:
            del DD[key]
    if remove_Rmax:
        BadData = []
        for key in DD.keys():
            Rmax = np.max(Calc_R(DD[key]['BT']))
            if  DD[key]['Rmidout'] - (Rmax+DeltaR) > 0:
                BadData.append(key)
                Error[key] = 'Pedestal not observed at time '+str(DD[key]['Time'])
        for key in BadData:
            del DD[key]
    return DD, Error    
    

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
These functions are used collect ECE, BT, and Density data at the requested time
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def Collect_ECE(Time,record):
    Data = np.zeros(40)
    for i, ece in enumerate(ECE_CHANNELS):
        ECE_Dat = record[ece]['data']
        ECE_Time = record[ece]['times']

        #find the time in ECE_Time closest to Time
        IDX = np.abs(ECE_Time - Time).argmin()  
        assert ECE_Time.shape == ECE_Dat.shape, 'Dimension of Data must match Dimension of Time'
        Data[i] = ECE_Dat[IDX]
    return Data

def Collect_Density(Time,record):
    Density_Time = record['DENSITY']['times']
    Density = record['DENSITY']['data']
    IDX = np.abs(Density_Time - Time).argmin()
    assert Density_Time.shape == Density.shape, 'Dimension of Data must match Dimension of Time'
    return Density[IDX]

def Collect_BT(Time,record):
    BT_Time = record['BT']['times']
    BT = record['BT']['data']
    IDX = np.abs(BT_Time - Time).argmin()
    assert BT_Time.shape == BT.shape, 'Dimension of Data must match Dimension of Time'
    return BT[IDX]

def Collect_Rmidout(Time,record):
    Rmidout_Time = record['Rmidout']['times']
    Rmidout = record['Rmidout']['data']
    IDX = np.abs(Rmidout_Time - Time).argmin()
    assert Rmidout_Time.shape == Rmidout.shape, 'Dimension of Data must match Dimension of Time'
    return Rmidout[IDX]

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
This function calculates the radial distances of the ECE chords
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def Calc_R(B_0):
    R_0 = 1.6955
    R = np.zeros(40)
    F1_16 = np.arange(83.5,98.5+1,1)
    F17_32 = np.arange(98.5,113.5+1,1)
    F33_40 = np.arange(115.5,129.5+1,1)
    F = np.concatenate((np.concatenate((F1_16,F17_32)),F33_40))

    for i in range(40):
        R[i] = 2*28*B_0*R_0/F[i]
    return abs(R)


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
Perform the RBF Fit and return the coefficients
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def calc_all_coefficients(DD,Error,RBFparam,CenterNum,include_00,maxtemp_loss):
    Dict_XY = {}
    
    Data = np.array([v['Data'] for v in DD.values()])
    Time = np.array([v['Time'] for v in DD.values()])
    BT = np.array([v['BT'] for v in DD.values()])
    Rmidout = np.array([v['Rmidout'] for v in DD.values()])
    
    for i in range(Data.shape[0]):
        try:
            X,Y = provide_xy(Data[i], Time[i],np.abs(BT[i]),Rmidout[i],maxtemp_loss,include_00)
            Weights, Centers = RBF_Fitter_Tool(Y,X,RBFparam,CenterNum)

            Dict_XY[i] = {'Weights':Weights,
                          'Time':Time[i]}
        except Exception as e:
            Error[Time[i]] = f'RBF fit failed at time {Time[i]}: {e}'
    return Dict_XY, Error

def provide_xy(Data,Time,BT,Rmidout,maxtemp_loss,include_00):
    R = Calc_R(BT)
    
    Y = Data[R <= Rmidout]
    X = R[R <= Rmidout]
    if include_00:
        X = np.concatenate((np.array([Rmidout]),X))
        Y = np.concatenate((np.array([0]),Y))

    #Remove data after cooling begins
    last_idx = np.where(Y > maxtemp_loss*np.max(Y))[0][-1]
    return X[:last_idx+1],Y[:last_idx+1]

def RBF_Fitter_Tool(Data,Radius,RBFparam,CenterNum):
    #Take in some (X,Y)/(Radius, Data) inputs and fit a series of radial basis functions to this data and output coefficients
    Centers = np.linspace(np.min(Radius),np.max(Radius),CenterNum)
    RBFmat = RBF_Gauss(Radius,Centers,RBFparam)

    Weights = np.linalg.lstsq(RBFmat, Data, rcond=None)[0]
    return Weights, Centers


def RBF_Gauss(X, Centers,RBFparam):
    diff = np.abs(X[:, None] - Centers[None, :])
    RBFmat = np.exp(-RBFparam*(diff**2))
    return RBFmat
