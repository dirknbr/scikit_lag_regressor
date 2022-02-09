"""
A scikit-learn style lagged regressor. It automatically finds the best lag

"""

import numpy as np
from sklearn import linear_model, metrics
# import pandas as pd
from itertools import product

def lag(x, l=1):
  y = np.roll(x, l)
  # TODO: when x is int, the nan can generate a number, fix it.
  y[:l] = np.nan
  return y

class LagRegressor:
  def __init__(self, base_reg=linear_model.LinearRegression, maxlag=4, 
  			   metric=metrics.mean_absolute_error, sample=500):
  	self.model = base_reg
  	self.maxlag = maxlag
  	self.metric = metric
  	self.lagset = None
  	self.sample = sample # max limit for random search

  def fit(self, X, y):
  	"""Fits the model.

  	Args:
  	  X: feature matrix, which will be lagged.
  	  y: target time series.
  	"""

  	# For each X series we find one optimal lag that minimises the `metric`.
  	n_feat = X.shape[1]
  	lagsets = [i for i in product(np.arange(self.maxlag + 1), repeat=n_feat)]
  	if len(lagsets) > self.sample:
  	  selected = np.random.choice(len(lagsets), self.sample, replace=False)
  	  lagsets = [lagsets[i] for i in selected]
  	print("Running", len(lagsets), "settings")
  	best = {'value': np.inf, 'lags': [], 'model': None}
  	for lagset in lagsets:
  	  Xlag = X.copy()
  	  for i in range(n_feat):
  	  	Xlag[:, i] = lag(Xlag[:, i], lagset[i])
  	  model = self.model()
  	  model.fit(Xlag[self.maxlag:, :], y[self.maxlag:])
  	  pred = model.predict(Xlag[self.maxlag:, :])
  	  error = self.metric(y[self.maxlag:], pred)
  	  if error < best['value']:
  	  	best = {'value': error, 'lags': lagset, 'model': model}
  	  	# print(error)
  	self.model = best['model']
  	self.lagset = best['lags']
  	print("Best model", best)


  	# TODO: add out of sample validation.
  	# TODO: add multiple lags per X series.

  def predict(self, X):
  	"""Predict target time series based on lagged X.

  	Args:
  	  X: feature matrix, which will be lagged.

  	Returns:
  	  future predictions
  	"""
  	assert X.shape[1] == len(self.lagset)
  	maxlag = max(self.lagset)
  	Xlag = X.copy()
  	for i in range(X.shape[1]):
  	  Xlag[:, i] = lag(Xlag[:, i], self.lagset[i])
  	pred = lag(np.arange(X.shape[0], dtype=np.float32), maxlag) # init with nan
  	pred[maxlag:] = self.model.predict(Xlag[maxlag:, :])
  	return pred


