import numpy as np

class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented


class Precision(object):    

    def __call__(self, predict, fit):
        tp = np.sum(fit * predict)
        fp = np.sum((fit == 0) * (predict == 1))
        return tp / (tp + fp)



class Recall(object):
    
    def __call__(self, predict, fit):
        tp = np.sum(fit * predict)
        fn = np.sum((fit == 1) * (predict == 0))
        return tp / (tp + fn)
    
    
    
class Accurency(object):
    
    def __call__(self, predict, fit):
        tp = np.sum(fit * predict)
        fp = np.sum((fit == 0) * (predict == 1))
        tn = np.sum((fit == 0) * (predict == 0))
        fn = np.sum((fit == 1) * (predict == 0))
        return (tp + tn) / (tp + tn + fn + fp)