import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .utils import vcol, vrow, evaluate_model

class SVM:
    def __init__(self, C=1.0, K=1.0, kernel='linear', d=2, c=0.0, gamma=0.1):
        self.C = C
        self.K = K
        self.kernel_type = kernel
        self.d = d
        self.c = c
        self.gamma = gamma

    def _compute_kernel(self, D1, D2):
        # D1: (f, n1), D2: (f, n2)
        if self.kernel_type == 'linear':
            return np.dot(D1.T, D2) 
        
        elif self.kernel_type == 'polinomial':
            return (np.dot(D1.T, D2) + self.c) ** self.d 
        
        elif self.kernel_type == 'rbf':
            # Fast RBF using: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x^Ty
            dist = np.sum(D1**2, axis=0).reshape(-1, 1) + \
                   np.sum(D2**2, axis=0) - 2 * np.dot(D1.T, D2)
            return np.exp(-self.gamma * dist) 
        
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def formulation_obj(self, alpha, H):
        # Dual objective: L_hat^D(alpha) = 0.5 * alpha^T * H * alpha - alpha^T * 1 
        ha = np.dot(H, alpha)
        loss = 0.5 * np.dot(alpha.T, ha) - np.sum(alpha) 
        grad = ha - np.ones(alpha.shape[0]) 
        return loss, grad

    def fit(self, D, L):
        self.x_train = D
        n_samples = D.shape[1]
        self.z = 2 * L - 1 # 

        # 1. Compute Kernel Matrix and add bias term xi = K^2 
        kernel_matrix = self._compute_kernel(D, D)
        G = kernel_matrix + (self.K ** 2) 

        # 2. Compute H matrix 
        self.H = np.outer(self.z, self.z) * G

        # 3. Optimize Dual 
        bounds = [(0, self.C)] * n_samples
        self.alpha, _, _ = fmin_l_bfgs_b(
            func=self.formulation_obj,
            x0=np.zeros(n_samples),
            args=(self.H,),
            bounds=bounds,
            factr=np.nan, # 
            pgtol=1e-5
        )
        return

    def predict(self, D_test):
        # Score computation: sum(alpha_i * z_i * k(x_i, x_t)) + xi 
        # k_test is (n_train, n_test)
        k_test = self._compute_kernel(self.x_train, D_test) + (self.K ** 2) 
        
        # scores = sum over training samples 
        scores = np.dot(self.alpha * self.z, k_test)
        self.pred = (scores > 0).astype(int)
        return self.pred

    def evaluate(self, y: np.ndarray):
        self.y_test = y

        err, cm = evaluate_model(self.y_test, self.pred, [0, 1])

        return (err, cm)