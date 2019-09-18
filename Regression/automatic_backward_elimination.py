# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Processing categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoinding the dummy variable trap
X = X[:, 1:]

#Build the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(np.ones(shape=(50,1)).astype(int), X, 1)

#Automatic Backward Elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_ABE = sm.OLS(y,x).fit()
        maxVar = max(regressor_ABE.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_ABE.pvalues[j].astype(float)==maxVar):
                    x = np.delete(x, j, 1)
    regressor_ABE.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#Testing if the model was correctly predicted
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled,y, test_size = 0.2,  random_state = 0)
from sklearn.linear_model import LinearRegression
regressorOLS = LinearRegression()
regressorOLS.fit(X_train, y_train)
y_pred = regressorOLS.predict(X_test)