import unittest
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import get_precision, get_recall, get_accuracy, get_fscore

class TestEvaluationMetrics(unittest.TestCase):

    def test_accuracy(self):
        y_pred = [1, 0, 1, 1, 0]
        y_true = [1, 0, 0, 1, 1]
        expected_accuracy = 0.6
        self.assertEqual(get_accuracy(y_pred, y_true), expected_accuracy)

    def test_get_precision(self):
        y_pred = [1,1,0,0,1,0,0,1,0,0]
        y_true = [1,1,1,1,1,0,0,0,0,0]
        label = 1
        expected_precision = 0.75
        self.assertEqual(get_precision(y_pred, y_true, label), expected_precision)

    def test_get_recall(self):
        y_pred = [1,1,0,0,1,0,0,1,0,0]
        y_true = [1,1,1,1,1,0,0,0,0,0] 
        label = 1
        expected_recall = 0.6  
        self.assertEqual(get_recall(y_pred, y_true, label), expected_recall)

if __name__ == '__main__':
    unittest.main()