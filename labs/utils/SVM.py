import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .utils import vcol, vrow, evaluate_model

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class SVM:
    def __init__(self, C: float = 1.0, K: float = 1.0, kernel: str = 'linear'):
        self.C = C
        self.K = K
        self.kernel = kernel

    def formulation_obj(self, alpha):
        # alpha is a 1-D array of shape (n,)
        # H is (n, n)
        
        # Dual Loss: 0.5 * alpha.T @ H @ alpha - sum(alpha) 
        ha = np.dot(self.H, alpha)
        loss = 0.5 * np.dot(alpha.T, ha) - alpha.sum()
        
        # Gradient: H @ alpha - 1 
        grad = ha - np.ones(alpha.shape[0])
        
        return loss, grad.flatten()

    def fit(self, D: np.ndarray, L: np.ndarray):
        # D is (f, n), L is (n,)
        n_samples = D.shape[1]
        self.z = 2 * L - 1 # Map {0, 1} to {-1, 1}

        # 1. Build extended feature matrix
        # x_ext becomes (f+1, n)
        row_K = np.ones((1, n_samples)) * self.K
        self.x_ext = np.vstack((D, row_K))

        # 2. Pre-compute H 
        G = np.dot(self.x_ext.T, self.x_ext)
        self.H = np.outer(self.z, self.z) * G

        # 3. Setup Optimization
        bounds = [(0, self.C)] * n_samples
        x0 = np.zeros(n_samples)

        alpha, min_loss, _ = fmin_l_bfgs_b(
            func=self.formulation_obj,
            x0=x0,
            bounds=bounds,
            factr=np.nan,
            pgtol=1e-5
        )

        # 4. Recover primal solution: w_hat* = sum(alpha_i * z_i * x_ext_i) 
        # (f+1, n) * (n,) -> weights for each extended feature
        self.w_ext = np.dot(self.x_ext, alpha * self.z)
        
        return min_loss

    def predict(self, D: np.ndarray):
        # D is (f, n_test)
        n_test = D.shape[1]
        D_ext = np.vstack((D, np.ones((1, n_test)) * self.K))
        
        # Score = w_ext.T @ D_ext [cite: 68]
        scores = np.dot(self.w_ext.T, D_ext)
        self.pred = (scores > 0).astype(int) # Map back to {0, 1} [cite: 70, 73]
        return self.pred

    def evaluate(self, y: np.ndarray):
        self.y_test = y

        err, cm = evaluate_model(self.y_test, self.pred, [0, 1])

        return (err, cm)