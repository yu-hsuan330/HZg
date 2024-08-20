# import ROOT
import uproot
import pandas as pd
import numpy as np
from dask import delayed
from scipy.interpolate import interp1d

import dask.dataframe as dd

def df_from_rootfiles(processes, treepath, branches, flatten='False', debug=False):
    
    def get_df(Class, file, wt, selection, treepath=None, branches="", multfactor=1):
        
        df = uproot.concatenate(file, branches, cut=selection, library="pd")
        if debug: df = df.head(10000)

        df["weight"] = 1.
        df["Class"] = Class

        # add weights
        if debug: print(type(wt))
        if type(wt) == type(('wei1','wei2')):
            for i in wt:
                df["weight"] *= df[i]
        elif type(wt) == type("hello"):
            df["weight"] = df[wt]
        elif (type(wt) == type(0.1)) or (type(wt) == type(1)):
            df["weight"] = wt
        else:
            print("CAUTION: weight should be a branch name or a number or a tuple... Assigning the weight as 1")   
        
        if debug: print(file, df["weight"].size, df["weight"].abs().sum())     
        
        return df

    dfs=[]
    for process in processes:
        if debug: print(process)

        if isinstance(process['path'], list):
            if isinstance(process['weight'], list):
                for onefile, onewt in zip(process['path'],process['weight']):
                    dfs.append(delayed(get_df)(process['Class'], onefile, process['weight'], process['selection'], treepath, branches))
            else:            
                for onefile in process['path']:
                    dfs.append(delayed(get_df)(process['Class'], onefile, process['weight'], process['selection'], treepath, branches))
        elif isinstance(process['path'], tuple) and len(process['path']) == 2:
            listoffiles=[process['path'][0]+'/'+f for f in os.listdir(process['path'][0]) if f.endswith(process['path'][1])]
            if debug: print(listoffiles)
            for onefile in listoffiles:
                dfs.append(delayed(get_df)(process['Class'], onefile, process['weight'], process['selection'], treepath, branches))
        elif isinstance(process['path'], str):
            dfs.append(delayed(get_df)(process['Class'], process['path'], process['weight'], process['selection'], treepath, branches))
        else:
            print("There is some problem with process path specification. Only string, list or tuple allowed")
    if debug: print("Creating dask graph!")
    if debug: print("Testing single file first")
    
    daskframe = dd.from_delayed(dfs)
    if debug: print("Finally, getting data from")
    output = daskframe.compute()
    output.reset_index(inplace = True, drop = True)
    return output
    
def df_balance_rwt(Mdf, SumWeightCol="instwei", NewWeightCol="NewWt", Classes=[""], debug=False) -> pd.Series:
    Mdf[NewWeightCol] = 1
    sum_w, sum_w_, wei = [1.0]*len(Classes), [1.0]*len(Classes), [1.0]*len(Classes)
    for i, k in enumerate(Classes):
        # calculate the weighted sum of each class
        sum_w[i] = Mdf[SumWeightCol][Mdf.Class == k].sum()
   
    # reweight the class[0](signal) only
    wei[0] = sum_w[1] / sum_w[0]
    wei[1] = 1

    for i, k in enumerate(Classes):    
        Mdf[NewWeightCol][Mdf.Class == k] = wei[i] * Mdf[SumWeightCol][Mdf.Class == k]
        sum_w_[i] = Mdf[NewWeightCol][Mdf.Class == k].sum()
        if debug: print("Class = %s, n = %.2f, balanced n = %.2f" %(k, sum_w[i], sum_w_[i]), flush=True)
    return Mdf[NewWeightCol]

