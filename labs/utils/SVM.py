import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .utils import vcol, vrow, evaluate_model

class SVM:
    def __init__(self, C=1.0, K=1.0, kernel='linear', d=2, c=0.0, gamma=0.1, costs=[[0, 1], [1, 0]], pi=[0.5, 0.5]):
        self.C = C
        self.K = K
        self.kernel_type = kernel
        self.d = d
        self.c = c
        self.gamma = gamma
        self.costs = costs
        self.pi = pi

    def _compute_kernel(self, D1, D2):
        # D1: (f, n1), D2: (f, n2)
        if self.kernel_type == 'linear':
            return np.dot(D1.T, D2) 
        
        elif self.kernel_type == 'polynomial':
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
        self.scores = np.dot(self.alpha * self.z, k_test)
        self.pred = (self.scores > 0).astype(int)
        return self.pred

    def calc_minDCF(self, y: np.ndarray):
        cur_pred = self.pred.copy()
        thresolds = np.sort(self.scores)
        first = thresolds[0]-1
        last = thresolds[-1]+1
        thresolds = thresolds[:-1] + np.diff(thresolds) / 2
        thresolds = np.concatenate(([first], thresolds, [last]))
        minDCF = 1000
        for t in thresolds:
            self.pred = np.where(self.scores >= t, 1, 0)
            _, _, _, DCF = self.evaluate(y)
            if DCF < minDCF: 
                minDCF = DCF

        self.pred = cur_pred

        return minDCF

    def evaluate(self, y: np.ndarray):
        self.y_test = y

        err, cm = evaluate_model(self.y_test, self.pred, [0, 1])

        Pfn = cm[0, 1] / (cm[0,1] + cm[1,1])
        Pfp = cm[1, 0] / (cm[1,0] + cm[0,0])
        Cfn = self.costs[0][1]
        Cfp = self.costs[1][0]
        pi1 = self.pi[1]

        DCFu = pi1*Cfn*Pfn + (1-pi1)*Cfp*Pfp
        DCF =  DCFu / min(pi1*Cfn, (1-pi1)*Cfp)

        return (err, cm, DCFu, DCF)