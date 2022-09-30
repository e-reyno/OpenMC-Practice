# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:55:42 2022

This file will process both the F_UNFOLD and the rate data from DIAMON spectrometer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path

class diamon_data():
     def __init__(self):
        """ define data"""
        self.file_name = ""
        self.name = 0
        self.dose_rate = 0
        self.dose_rate_uncert = None
        self.dose_area_product = 0
        self.dose_area_product_uncert = None
        self.thermal = 0
        self.epi = 0
        self.fast = 0
        self.energy_bins = []
        self.flux_bins = []
        self.count_D1 = 0
        self.count_D2 = 0
        self.count_D3 = 0
        self.count_D4 = 0
        self.count_D5 = 0
        self.count_D6 = 0
        self.count_F = 0
        self.count_FL = 0
        self.count_FR = 0
        self.count_R = 0
        self.count_RR = 0
        self.count_RL = 0
        self.count_time = 0
        self.phi = 0
        self.phi_uncert = None
        
        
#set of functions for string cleaning

def clean_param(line, uncert=None):
    line = line.split(":")[1]
    line = clean(line)
    
    if uncert:
        #uses regular expression package to extract an uncertainty found between a bracket and % symbol
        line[3] = re.findall('\((.*?)%\)', line[3])
        return float(line[0]), float(line[3][0])
    else:
        return float(line[0])

def clean_counts(line):
    line = clean(line)
    return int(line[1]), int(line[3])

def clean(line):
    line = line.strip()
    line = " ".join(line.split())
    line = line.split()
    return line

def read_data_file(path, i, j):
    
    data = pd.read_csv(path, sep='\t', index_col=False)
    data = data.dropna(axis='columns')
    data = data.drop(data.iloc[:, i:j], axis=1)
    data = data.replace('\%', '', regex=True)
    for col in data.columns:
        if 'un%' in col:
            data[col]= data[col].astype(float)
    return data

    
def read_unfold_file(path):

    results = diamon_data()
    results.file_name = Path(path).stem
    in_spect = False
    with open(path) as f:

        for line in f:

            if " thermal" in line:
                results.thermal= clean_param(line)
            elif "epi" in line:
                results.epi = clean_param(line)
            elif "fast" in line:
                results.fast = clean_param(line)
            elif "phi" in line:
                results.phi, results.phi_uncert = clean_param(line, True)
            elif "H*(10)_r" in line:
                results.dose_rate, results.dose_rate_uncert = clean_param(line, True)
            elif "h*(10)" in line:
                results.dose_area_product, results.dose_area_product_uncert = clean_param(line, True)
            elif "D1" in line:
                results.count_D1, results.count_R = clean_counts(line)
            elif "D2" in line:
                results.count_D2, results.count_RL = clean_counts(line)
            elif "D3" in line:
                results.count_D3, results.count_FL = clean_counts(line)
            elif "D4" in line:
                results.count_D4, results.count_F = clean_counts(line)
            elif "D5" in line:
                results.count_D5, results.count_FR = clean_counts(line)
            elif "D6" in line:
                results.count_D6, results.count_RR = clean_counts(line)
            elif "TIME" in line:
                line = clean_param(line)
                results.count_time = float(line)
            elif in_spect and ("----" not in line):
                line = clean(line)
                if len(line)<1:
                    break
                results.energy_bins.append(float(line[0]))
                results.flux_bins.append(float(line[-1]))
        
            elif "Ec" and "Phi(E)*Ec" in line:
                in_spect = True
    f.close()
    return results


def read_folder(folder_path):

    
    
    folder_list = glob.glob(folder_path)
    data = [[] for i in range(len(folder_list))]
    for i, folder in enumerate(folder_list):
        
        files_list = glob.glob(folder+"\*")
        unfold_data = read_unfold_file(files_list[1])
        unfold_data.name = folder.split("\\")[-1]
        #out data gives the dose and %energy neutrons as func of time in set time intervals (6 measurements)
        out_data = read_data_file(files_list[2],0,3)
        #rate data gives all counts as func of time for each detector in set time intervals (6 measurements)
        rate_data = read_data_file(files_list[3],0,1)
       
        data[i].append(unfold_data)
        data[i].append(out_data)
        data[i].append(rate_data)

    return data

    
def convert_to_ds(data):
    labels = ("file_name", "name", "dose_rate","dose_rate_uncert", "dose_area_product", "dose_area_prod_uncert", "thermal","epi", "fast", "phi", "phi_uncert",
              "D1", "D2", "D3", "D4", "D5", "D6", "F", "FL", "FR", "R", "RR", "RL", "time")
    
    data_list = [data.file_name, data.name, data.dose_rate, data.dose_rate_uncert, data.dose_area_product, data.dose_area_product_uncert, data.thermal, data.epi, data.fast, 
                data.phi, data.phi_uncert, data.count_D1, data.count_D2, data.count_D3, data.count_D4, data.count_D5, data.count_D6,
                data.count_F, data.count_FL, data.count_FR, data.count_R, data.count_RR, data.count_RL, data.count_time]
    
    s1 = pd.Series(data_list, index=labels)
    
    return s1


def combine_continuous_data_files(dataframes, cum_time=None):
    combined_dataframe = []
    for i, dataframe in enumerate(dataframes):

        if cum_time and i != 0:

            last_index = dataframes[i-1].iloc[-1,0]
            #this aligns the new files time with the previous so they are adjacent
            dataframe.iloc[:,0] = last_index + dataframe.iloc[:,0]

        combined_dataframe.append(dataframe)
    combined_dataframe = pd.concat(combined_dataframe, ignore_index=True)

    return combined_dataframe
