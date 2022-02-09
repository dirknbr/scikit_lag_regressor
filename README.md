# Scikit-learn regressor with lagging of features

We build a simple sklearn regressor that can use linear or other models.
It does a search for the best lags of X.


```
n = 50
X = np.random.normal(10, 1, size=(n, 4))
y = 10 + lag(X[:, 0], 2) + lag(X[:, 1], 3) + np.random.normal(0, 2, n)

reg = LagRegressor()
reg.fit(X, y)
pred = reg.predict(X)
print(pred[:10])

```
