from utils.utils import get_cov, vcol, vrow, logpdf_GAU_ND, get_class_covariances
import numpy as np
from scipy.special import logsumexp

##################################### Helper Classes ##################################

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
        self.P = vcol(P)

    def evaluate(self, y):
        self.y_test = y

        acc = np.sum(self.y_test==self.pred) / len(self.y_test)
        err = 1 - acc

        return (err)

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
        self.t = np.log(self.P[1] / self.P[0])

class MultiClass(BayesClassifier):
    def predict(self, x: np.ndarray):
        self.calc_logS(x)

        logPc = np.log(self.P)
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
