import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df_train0 = pd.read_csv("train.csv")
train2 = df_train0.loc[0:df_train0.shape[0]/5]
test2 = df_train0.loc[df_train0.shape[0]/5+1:df_train0.shape[0]-1]


test2result = pd.read_csv("output.csv")
test2result = test2result.Prediction.values
test2expect = test2.gap.values

print test2expect.shape
print test2result.shape

diffsq = np.square(test2result - test2expect)
RMSE = math.sqrt(np.sum(diffsq)/diffsq.shape[0])

print RMSE

