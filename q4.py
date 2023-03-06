import numpy as np
import pandas as pd

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
