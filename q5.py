import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statistics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = pd.read_csv("country_vaccination_stats.csv")

median_list = []
k = []
for j in range(0, len(data) - 1):
    
    if data["country"][j] == data["country"][j + 1]:
        k.append(data["daily_vaccinations"][j + 1])
        if j == len(data) - 2:
            if k != []:
                median_list.append([statistics.median(k),data["country"][j]])
            break
    else:
        if k != []:
            median_list.append([statistics.median(k),data["country"][j]])
        k = []
        
df = pd.DataFrame(median_list, columns = ['median', 'country'])
df_new = df.sort_values(by = ['median'], ascending = False)
print(df_new[0:3])
