import numpy as np

class Bspline():
    def __init__(self):
        self.fitted = {}
        self.settings = {}

    def B(self, x, i, m, t):
        '''
        ith B-Spline basis function of order m.
        :param x: 1d input
        :param i: spline function index
        :param m: degree of spline
        :param t: knot sequence
        '''
        self.tau = t
        if m == 0:
            return 1.0 if t[i] <= x <= t[i + 1] else 0.0
        c1, c2 = 0.0, 0.0
        if t[i + m] > t[i]:
            c1 = (x - t[i]) / (t[i + m] - t[i]) * self.B(x, i, m - 1, t)
        if t[i + m + 1] > t[i + 1]:
            c2 = (t[i + m + 1] - x) / (t[i + m + 1] - t[i + 1]) * self.B(x, i + 1, m - 1, t)
        return c1 + c2

    def build_spline_mat(self, data, K, M, t):
        '''
        Evaluate Pointwise Spline Matrix
        '''
        X = np.zeros(shape=(data.shape[0], K + M))
        for n in range(K + M):
            for i in range(X.shape[0]):
                X[i, n] = self.B(data[i], n, M - 1, t)
        return X

    def build_knot_vec(self, knots, M, x):
        '''
        build Open Uniform knot vector

        extension: duplicate knots
        '''
        return np.array([x.min()] * M + knots + [x.max()] * M)

    def fit(self, x, y, M, knots):
        '''
        least squares of spline coefficients on data.
        :param M: order of bspline (degree+1)
        :param knots: interior knots
        :param s: smoothing/regularisation parameter
        '''
        tau = self.build_knot_vec(knots, M, x)
        K = len(knots)

        X = self.build_spline_mat(x, K, M, tau)
        b = np.linalg.inv(X.T @ X) @ X.T @ y
        y_hat = X @ b
        sigma_sq_hat = np.mean((y - y_hat) ** 2)

        self.settings['tau'] = self.tau
        self.settings['M'] = M
        self.settings['K'] = K

        self.fitted['X'] = X
        self.fitted['b'] = b
        self.fitted['y_hat'] = y_hat
        self.fitted['sigma_sq_hat'] = np.mean((y - y_hat) ** 2)
        self.fitted['b_var_hat'] = np.linalg.inv(X.T @ X) * sigma_sq_hat

    def predict(self, x):
        h_x = self.build_spline_mat(x, self.settings['K'], self.settings['M'], self.settings['tau'])
        return h_x @ self.fitted['b'], self.standard_err(h_x)

    def standard_err(self, x):
        '''
        Classical Confidence interval calculation
        '''
        se = np.zeros(shape=(x.shape[0]))
        for i in range(x.shape[0]):
            se[i] = np.sqrt(x[i] @ np.linalg.inv(self.fitted['X'].T @ self.fitted['X']) @ x[i]) \
                    * np.sqrt(self.fitted['sigma_sq_hat'])
        return se











