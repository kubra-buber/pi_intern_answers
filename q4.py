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

data.fillna(0, inplace = True)
temp_data = []
first_index = 0
for j in range(0, len(data)-1):
    
    if data["country"][j] == data["country"][j + 1]:
        temp_data.append(data["daily_vaccinations"][j + 1])
    if data["country"][j] != data["country"][j + 1] or j == len(data) - 2:
        if temp_data != []:
            data.at[first_index, "daily_vaccinations"] = min(temp_data)
        temp_data = []
        first_index = j + 1
