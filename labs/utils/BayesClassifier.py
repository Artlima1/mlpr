from utils.utils import get_cov, vcol, vrow, logpdf_GAU_ND, get_class_covariances, evaluate_model
import numpy as np
from scipy.special import logsumexp

##################################### Helper Classes and functions ##################################

class GaussianDist:
    def __init__(self, D: np.ndarray, independent = False, cov=None, mu=None) -> None:
        if mu is None:
            self.mu = vcol(D.mean(1))
        else:
            self.mu = mu.copy()

        if cov is None:
            if not independent:
                self.cov = get_cov(D)
            else: 
                cov = get_cov(D)
                self.cov = cov * np.eye(cov.shape[0], cov.shape[1])
        else:
            self.cov = cov.copy()

######################################## Base Bayes Classifier ################################

class BayesClassifier:

    def __init__(self):
        self.x_train = np.zeros(1)
        self.y_train  = np.zeros(1)
        self.x_test = np.zeros(1)
        self.y_test  = np.zeros(1)
        self.labels = []
        self.class_dists = {}
        self.pred = np.zeros(1)
        self.priors = np.zeros(1)
        self.logS = np.zeros(1)
        self.C = np.zeros(1)


    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x_train = x
        self.y_train = y
        self.labels = np.unique(y).tolist()

    def calc_logS(self, x: np.ndarray):
        self.x_test = x

        lls = []
        for l in self.labels:
            lls.append(
                logpdf_GAU_ND(
                    self.x_test, 
                    self.class_dists[l].mu, 
                    self.class_dists[l].cov
                )
            )

        self.logS = np.vstack(lls)

    def set_priors(self, P: np.ndarray):
        self.pi = vcol(P.copy())

    def evaluate(self, y:np.ndarray):
        self.y_test = y

        return evaluate_model(self.y_test, self.pred, self.labels)

    def set_cost_matrix(self, C):
        self.C = C

    def predict(self, x: np.ndarray):
        raise NotImplementedError("Use Binary or MultiClass subclass for predict().")

################################### Gaussian Fitting Method ##################################

class MVG(BayesClassifier):
    def fit(self, x: np.ndarray, y: np.ndarray):
        super().fit(x, y)

        for l in self.labels:
            self.class_dists[l] = GaussianDist(self.x_train[:, self.y_train==l])
            

class NaiveBayes(BayesClassifier):
    def fit(self, x: np.ndarray, y: np.ndarray):
        super().fit(x, y)

        for l in self.labels:
            self.class_dists[l] = GaussianDist(self.x_train[:, self.y_train==l], independent=True)


class TiedVariance(BayesClassifier):
    def fit(self, x: np.ndarray, y: np.ndarray):
        super().fit(x, y)

        _, Sw = get_class_covariances(self.x_train, self.y_train, self.labels)

        for l in self.labels:
            self.class_dists[l] = GaussianDist(self.x_train[:, self.y_train==l], cov=Sw)

##################################### Binary vs Multiclass ##################################

class Binary(BayesClassifier):
    def predict(self, x: np.ndarray):
        self.calc_logS(x)

        self.llr = self.logS[1] - self.logS[0]

        self.pred = np.where(self.llr >= self.t, 1, 0)

    def set_threshold_via_prior_ratio(self):
        self.t = np.log(self.pi[1] / self.pi[0])

    def set_optimal_thresold(self):
        Cfn = self.C[0, 1]
        Cfp = self.C[1, 0]
        pi_1 = self.pi[1]
        self.t = -np.log((pi_1 * Cfn) / ((1-pi_1) * Cfp))

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

    def evaluate(self, y: np.ndarray): # type: ignore
        # TODO: DCFu can also be generalized to however many classes

        err, cm = super().evaluate(y)

        Pfn = cm[0, 1] / (cm[0,1] + cm[1,1])
        Pfp = cm[1, 0] / (cm[1,0] + cm[0,0])
        Cfn = self.C[0, 1]
        Cfp = self.C[1, 0]
        pi1 = self.pi[1, 0]

        DCFu = pi1*Cfn*Pfn + (1-pi1)*Cfp*Pfp
        DCF =  DCFu / min(pi1*Cfn, (1-pi1)*Cfp)

        return (err, cm, DCFu, DCF)
    
    
    

class MultiClass(BayesClassifier):
    def predict(self, x: np.ndarray):
        self.calc_logS(x)

        logPc = np.log(self.pi)
        logSJoint = self.logS + logPc
        logSMarginal = vrow(logsumexp(logSJoint, axis=0)) # type: ignore
        logSPost = logSJoint - logSMarginal
        SPost = np.exp(logSPost)

        self.pred = np.argmax(SPost, axis=0)


##################################### Actual Classifier Instances ##################################

class BinaryMVG(MVG, Binary):
    pass

class MultiClassMVG(MVG, MultiClass):
    pass

class BinaryNaiveBayes(NaiveBayes, Binary):
    pass

class MultiClassNaiveBayes(NaiveBayes, MultiClass):
    pass

class BinaryTiedVariance(TiedVariance, Binary):
    pass

class MultiClassTiedVariance(TiedVariance, MultiClass):
    pass
