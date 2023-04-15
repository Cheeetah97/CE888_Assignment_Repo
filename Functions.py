import pandas as pd
from datetime import datetime,timedelta
import numpy as np
from scipy.stats import kurtosis,skew,entropy
import collections
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Data Combining Function

def combine(file,filename):
    
    # Removing Extra Rows from the Files
    start_timestamp = file.iloc[0,0]
    sample_rate = file.iloc[1,0]
    file = file[2:]
    file = file.reset_index(drop=True)
    
    # Renaming the Columns
    if len(file.columns) > 1:
        file.columns = [filename+"_X",filename+"_Y",filename+"_Z"]
    else:
        file.columns = [filename]
    
    # Converting Timestamp to Datetime
    file["Datetime"] = datetime.fromtimestamp(start_timestamp)
    
    start_date = file["Datetime"].iloc[0] - timedelta(seconds = 1/sample_rate)
    end_date = file["Datetime"].iloc[0] + timedelta(seconds = (len(file)-2)*(1/sample_rate))
    file["Datetime"] = pd.date_range(start=start_date,end=end_date,freq=f'{1/sample_rate}S') + pd.Timedelta(seconds = 1/sample_rate)
    
    # Resampling the Data to 4Hz
    cols = [i for i in file.columns if i not in ["Datetime"]]
    if sample_rate > 4:
        file.index = file["Datetime"]
        file = file.drop("Datetime",axis=1)
        for col in cols:
            file[col] = file[col].resample('0.25S').mean()
        file = file.reset_index().rename(columns={"index":"Datetime"})
        file = file.dropna()  
    return file

#%%
# Statistics Calculation Funtion

def calculate_statistics(list_values):
    
    # Removing NaNs from List
    list_values = [val for val in list_values if str(val) != 'nan']
    
    # Median
    median = np.nanpercentile(list_values,50)
    
    # Mean
    mean = np.nanmean(list_values)
    
    # Standard Deviation
    std = np.nanstd(list_values)
    
    # Kurtosis
    kurt = kurtosis(list_values)
    
    # Skewness
    skewness = skew(list_values)
    
    # Mean Crossings
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values)>np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    
    # Entropy
    counter_values = collections.Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entr = entropy(probabilities)
    
    return [median,mean,std,kurt,skewness,no_mean_crossings,entr]

#%%
# ReSampling Function

def re_sample(df):
    stress = df[df["Stress level"] == 1.0]
    no_stress  = df[df["Stress level"] == 0.0]
    df_downsample = resample(stress,replace=True,n_samples=50,random_state=42)
    df_upsample = resample(no_stress,replace=True,n_samples=50,random_state=42)
    data = pd.concat([df_upsample,df_downsample])
    return data.reset_index(drop=True)

#%%
# Confusion Matrix Function

def plot_confusion_matrix(actual_classes,predicted_classes,sorted_labels):

    matrix = confusion_matrix(actual_classes,predicted_classes,labels=sorted_labels)
    
    plt.figure(figsize=(10,5))
    sns.heatmap(matrix,annot=True,xticklabels=sorted_labels,yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()
