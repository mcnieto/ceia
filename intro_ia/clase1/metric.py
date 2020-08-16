import numpy as np

class BaseMetric(self):
    def __int__(self, prediction, truth):
        self.prediction = prediction
        self.truth = truth

class precision(BaseMetric):
    def __call__(self, prediction, truth):
        tp = np.sum(truth * prediction)
        fp = np.sum((truth == 0) * (prediction == 1))
        return tp / (tp + fp)

class recall(BaseMetric):
    def __call__(self, prediction, truth):
        tp = np.sum(truth * prediction)
        fn = np.sum((truth == 1) * (prediction == 0))
        return tp / (tp + fn)    
    
class accurency(BaseMetric):
    def __call__(self, prediction, truth):
        tp = np.sum(truth * prediction)
        fp = np.sum((truth == 0) * (prediction == 1))
        tn = np.sum((truth == 0) * (prediction == 0))
        fn = np.sum((truth == 1) * (prediction == 0))
        return (tp + tn) / (tp + tn + fn + fp)