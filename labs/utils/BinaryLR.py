import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .utils import vcol, vrow, evaluate_model

class BinaryLR:
    def __init__(self, lamb: float = 1e-3, C: np.ndarray = np.array([[0, 1], [1, 0]]), pi1: float = 0.5):
        self.lamb = lamb
        self.C = C
        self.pi = vcol(np.array([(1-pi1), pi1]))

    def set_lambda(self, lamb: float):
        self.lamb = lamb

    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]

        J = (self.lamb / 2) * np.linalg.norm(w)**2

        z = 2*self.y_train - 1

        S = np.dot(vcol(w).T, self.x_train).ravel() + b
        G = -z / (1 + np.exp(z * S))

        if self.prior_weighted:
            eps = np.zeros(z.shape)
            eps0 = self.pi[1, 0] / np.sum(self.y_train == 1)
            eps1 = self.pi[0, 0] / np.sum(self.y_train == 0)
            eps[z == 1] = eps0
            eps[z == -1] = eps1

            J += np.sum(eps*np.logaddexp(0, -z * S))
            
            grad_w = np.sum((vrow(eps*G) * self.x_train), axis=1) + self.lamb * w.ravel()
            grad_b = np.sum(eps * G)
        else:
            J += np.logaddexp(0, -z * S).mean()
            grad_w = (vrow(G) * self.x_train).mean(1) + self.lamb * w.ravel()
            grad_b = G.mean()

        grad = np.hstack((grad_w, grad_b))
        
        return J, grad

    def fit(self, D: np.ndarray, L: np.ndarray, prior_weighted = False):
        self.x_train = D
        self.y_train = L
        self.prior_weighted = prior_weighted
        x0 = np.zeros(D.shape[0] + 1)
        opt_v, opt_loss, d  = fmin_l_bfgs_b(func=self.logreg_obj, x0=x0, approx_grad=False)
        self.opt_loss = opt_loss
        self.w = opt_v[0:-1]
        self.b = opt_v[-1]
        return opt_loss

    def predict(self, D: np.ndarray) -> np.ndarray:
        S = self.w.T @ D + self.b
        self.pred = (S > 0).astype(int)
        self.llr = S - np.log( self.pi[1, 0] / self.pi[0, 0] )
        return self.pred


    # TODO: generalize for both classifiers develped
    def calc_minDCF(self, y: np.ndarray):
        cur_pred = self.pred.copy()
        thresolds = np.sort(self.llr)
        first = thresolds[0]-1
        last = thresolds[-1]+1
        thresolds = thresolds[:-1] + np.diff(thresolds) / 2
        thresolds = np.concatenate(([first], thresolds, [last]))
        minDCF = 1000
        for t in thresolds:
            self.pred = np.where(self.llr >= t, 1, 0)
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
        Cfn = self.C[0, 1]
        Cfp = self.C[1, 0]
        pi1 = self.pi[1, 0]

        DCFu = pi1*Cfn*Pfn + (1-pi1)*Cfp*Pfp
        DCF =  DCFu / min(pi1*Cfn, (1-pi1)*Cfp)

        return (err, cm, DCFu, DCF)