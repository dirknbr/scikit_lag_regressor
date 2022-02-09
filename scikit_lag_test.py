
import unittest
import numpy as np
from scikit_lag import lag, LagRegressor

class TestLagRgressor(unittest.TestCase):

  def test_lag_nan(self):
  	x = np.arange(10, dtype=np.float32)
  	lagged = lag(x, 2)
  	self.assertEqual(lagged[-1], 7)
  	self.assertTrue(np.isnan(lagged[1]))

  def test_regressor(self):
  	np.random.seed(22)
  	n = 50
  	X = np.random.normal(0, 1, size=(n, 2))
  	y = np.random.normal(0, 1, n)
  	reg = LagRegressor(maxlag=2)
  	reg.fit(X, y)
  	pred = reg.predict(X)
  	self.assertEqual(len(pred), n)


if __name__ == '__main__':
  unittest.main()