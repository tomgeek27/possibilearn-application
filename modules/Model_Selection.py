from possibilearn import *
from tesimodules import Validation
import numpy as np
from multiprocessing import Pool

def Get_mu(dataset_labels):
    mu = {}
    classes = np.unique(dataset_labels)

    for clas in classes:
        mu[clas] = [1 if label == clas else 0 for label in dataset_labels]
    
    return mu

class Learn:
    def __init__(self, dataset_values, dataset_labels):
        self.dataset_values = dataset_values
        self.dataset_labels = dataset_labels
        self.dimension = len(dataset_values[0])

        self.mu = Get_mu(dataset_labels)
    
    def __get_generator(self):
        d = self.dimension

        return lambda m: (-4 + np.random.random(d*m) * 8).reshape((m, d))
        return (-4 + np.random.random(2*m) * 8).reshape((m, 2))


class Holdout(Learn):

    def __init__(self, dataset_values, dataset_labels, random_state=None, size_train=0.6, size_validation=0.2):
        super().__init__(dataset_values, dataset_labels)
        self.generate_train_validation_test_sets(random_state=random_state, size_train=size_train, size_validation=size_validation)
        
    def generate_train_validation_test_sets(self, random_state, size_train, size_validation):
        self.train_values, self.train_labels, self.validation_values, self.validation_labels, self.test_values, self.test_labels, self.train_indices, self.validation_indices, self.test_indices = Validation.split_train_validation_test_dataset(dataset_values=self.dataset_values, dataset_labels=self.dataset_labels, random_state=random_state, size_train=size_train, size_validation=size_validation)

    def get_mu_train(self):
        return self.mu_train

    def get_values_train(self):
        return self.train_values

    def get_values_validation(self):
        return self.validation_values 

    def get_values_test(self):
        return self.test_values

    def get_labels_train(self):
        return self.train_labels
    
    def get_labels_validation(self):
        return self.validation_labels
    
    def get_labels_test(self):
        return self.test_labels

    def get_indices_train(self):
        return self.train_indices
    
    def get_indices_validation(self):
        return self.validation_indices

    def get_indices_test(self):
        return self.test_indices

    def get_values_sets(self):
        return (self.get_values_train(), self.get_values_validation(), self.get_values_test())

    def get_labels_sets(self):
        return (self.get_labels_train(), self.get_labels_validation(), self.get_labels_test())

    def get_labels_indexed_train(self):
        return list(zip(self.get_indices_train(), self.get_labels_train()))

    def get_labels_indexed_validation(self):
        return list(zip(self.get_indices_validation(), self.get_labels_validation()))

    def get_labels_indexed_test(self):
        return list(zip(self.get_indices_test(), self.get_labels_test()))

    def merge_values_train_validation(self):
        return self.get_values_train() + self.get_values_validation()
    
    def merge_labels_train_validation(self):
        return self.get_labels_train() + self.get_labels_validation()
    
    def merge_labels_indexed_train_validation(self):
        return self.get_labels_indexed_train() + self.get_labels_indexed_validation()