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

df = pd.read_csv("pi_intern_dataset.csv")

f1 = df["feature_1"]
f2 = df["feature_2"]
f3 = df["feature_3"]
f4 = df["feature_4"]
virus = df["isVirus"]

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f1, f2, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f1", fontsize = 20)
plt.ylabel("f2", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f1, f3, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f1", fontsize = 20)
plt.ylabel("f3", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f1, f4, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f1", fontsize = 20)
plt.ylabel("f4", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f2, f3, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f2", fontsize = 20)
plt.ylabel("f3", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f2, f4, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f2", fontsize = 20)
plt.ylabel("f4", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f3, f4, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f3", fontsize = 20)
plt.ylabel("f4", fontsize = 20)
plt.show()

df = df.dropna().reset_index(drop = True)

f1 = df["feature_1"]
f2 = df["feature_2"]
f3 = df["feature_3"]
f4 = df["feature_4"]
virus = df["isVirus"]

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f1, f2, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f1", fontsize = 20)
plt.ylabel("f2", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f1, f3, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f1", fontsize = 20)
plt.ylabel("f3", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f1, f4, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f1", fontsize = 20)
plt.ylabel("f4", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f2, f3, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f2", fontsize = 20)
plt.ylabel("f3", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f2, f4, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f2", fontsize = 20)
plt.ylabel("f4", fontsize = 20)
plt.show()

plt.figure(figsize = (11, 5), dpi = 80)
plt.scatter(f3, f4, c = virus, cmap = plt.cool(), label = "Virus")
plt.legend(loc = 'upper right', fontsize = 15)
plt.grid(True)
plt.xlim(-8, 8)
plt.ylim(-8, 10)
plt.xlabel("f3", fontsize = 20)
plt.ylabel("f4", fontsize = 20)
plt.show()

X = []
for i in range(0, len(df)):
    X.append([df["feature_1"][i], df["feature_2"][i], df["feature_3"][i], df["feature_4"][i]])
    
Y = []
for i in range(0, len(df)):
    Y.append(df["isVirus"][i])
    
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1, stratify = Y)

model = LogisticRegression(solver = 'liblinear', random_state = 0)
model.fit(X_train, Y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_pred = model.predict(X_test)

print(model.intercept_, model.coef_, model.score(X_train, Y_train))
print(model.intercept_, model.coef_, model.score(X_test, Y_test))

cm = confusion_matrix(Y_test, model.predict(X_test))

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks = (0, 1), ticklabels = ('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks = (0, 1), ticklabels = ('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha = 'center', va ='center', color = 'red')
plt.show()

print(classification_report(Y_test, model.predict(X_test)))
print(classification_report(Y_train, model.predict(X_train)))
