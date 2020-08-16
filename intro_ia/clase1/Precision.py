import numpy as np

class Precision(object):
    
    def __call__(self, prediction, truth):
        tp = np.sum(truth * prediction)
        fp = np.sum((truth == 0) * (prediction == 1))
        return tp / (tp + fp)



class Recall(object):
    
    def __call__(self, prediction, truth):
        tp = np.sum(truth * prediction)
        fn = np.sum((truth == 1) * (prediction == 0))
        return tp / (tp + fn)
    
    
    
class Accurency(object):
    
    def __call__(self, prediction, truth):
        tp = np.sum(truth * prediction)
        fp = np.sum((truth == 0) * (prediction == 1))
        tn = np.sum((truth == 0) * (prediction == 0))
        fn = np.sum((truth == 1) * (prediction == 0))
        return (tp + tn) / (tp + tn + fn + fp)