import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression

oj_og = pd.read_csv("https://raw.githubusercontent.com/makbigc/ISLR/refs/heads/master/datasets/OJ.csv")
oj_og.columns = oj_og.columns.str.replace(' ', '_').str.lower()  # Clean column names

oj = oj_og.copy()
oj['q'] = np.exp(oj['logmove'])
oj['weighted_mean'] = oj.groupby(['store', 'week'])['price'].apply(lambda x: np.average(x, weights=oj.loc[x.index, 'q']))
