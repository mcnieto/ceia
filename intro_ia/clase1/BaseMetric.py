class BaseMetric(self):
    def __int__(self, prediction, truth):
        self.prediction = prediction
        self.truth = truth
        
    """
    Definition:
    Is the number of true positives (TP) divided by the sum of true positives (TP) and false positives (FP)
    Example:
    prediction = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
    truth = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
    tp = 2
    fp = 2
    tp + fp = 4
    precision = 0.5
    """
        true_pos_mask = (prediction == 1) & (truth == 1)
        true_pos = true_pos_mask.sum()
        false_pos_mask = (prediction == 1) & (truth == 0)
        false_pos = false_pos_mask.sum()
        return true_pos / (true_pos + false_pos)