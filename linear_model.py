import numpy as np


class LinearRegression(object):
    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

        self._coef = None
        self._intercept = None
        self._new_X = None

    def fit(self, X, y):
        
        if self.fit_intercept == True :
            bias_vector = np.ones((np.size(X), 1))
            X = np.append(bias_vector, X, axis=1)
            
        X_transpose = np.transpose(X)
        X_transpose_dot_x = X_transpose.dot(X)
        X_inv = np.linalg.inv(X_transpose_dot_x)
        y_temp = X_transpose.dot(y)  
        theta = X_inv.dot(y_temp)
        self._coef = theta
        self._intercept = theta[1]

    def predict(self, X):
        return X.dot(self._intercept) + self._coef[0]


    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept
