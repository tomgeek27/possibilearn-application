from tesimodules import Validation
from tesimodules.Model_Selection import *
import numpy as np
import unittest

class ValidationTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ValidationTest, self).__init__(*args, **kwargs)
        self.dataset_values = list(range(10))
        self.dataset_labels = list(map(str, range(10)))

    def test_get_best_performance_with_multiple_elements(self):
        perfomance = {
            (1, 0.1): 0.9,
            (1, 0.2): 0.92,
            (10, .5): 1.0
        }

        self.assertEqual((10, .5), Validation.get_best_performance(performances=perfomance))

    def test_get_best_performance_with_array(self):
        performance = [1,2,4]

        self.assertRaises(Exception, Validation.get_best_performance, performance)

    def test_get_best_performance_with_no_elements(self):
        performances = {}
        self.assertRaises(Exception, Validation.get_best_performance, performances)
    
    def test_split_dataset(self):

        train_values, train_labels, validation_values, validation_labels, test_values, test_labels, train_indices, validation_indices, test_indices = Validation.split_dataset(self.dataset_values, self.dataset_labels)

        self.assertEqual(len(train_values), 6)
        self.assertEqual(len(train_labels), 6)

        self.assertEqual(len(validation_values), 2)
        self.assertEqual(len(validation_labels), 2)

        self.assertEqual(len(test_values), 2)
        self.assertEqual(len(test_labels), 2)

class ModelTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ModelTest, self).__init__(*args, **kwargs)

        membership_functions = {
            '0': lambda x: 0,
            '0.5': lambda x: 0.5,
            '1': lambda x: 1
        }

        self.model = Model_PL(1, 1, membership_functions)

    def test_score(self):
        values = [1,2,3,4]
        labels_one = ['1', '1', '1', '1']
        labels_half_one = ['1', '1', '0', '0']
        self.assertEqual(1.0, (self.model.score(values, labels_one)))
        self.assertEqual(0.5, (self.model.score(values, labels_half_one)))
    
    def test_score_with_no_labels(self):
        values = [1,2,3,4]
        labels = []
        self.assertRaises(AssertionError, self.model.score, values, labels)


if __name__=='__main__':
    unittest.main()