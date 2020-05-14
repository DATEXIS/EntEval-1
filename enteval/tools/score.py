from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

class Score(object):
    @classmethod
    def empty(cls):
        return Score(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_data(cls, y_true, y_pred, average='macro'):
        logging.debug("y_true.shape: %s, y_pred.shape: %s", y_true.shape, y_pred.shape)
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, beta=1, average=average)
        return Score(accuracy, precision, recall, f1)

    def __init__(self, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def get_accuracy(self):
        return self.accuracy

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return self.f1

    def __repr__(self):
        return "Score(accuracy: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%)" % (self.accuracy * 100, self.precision * 100, self.recall * 100, self.f1 * 100)
