from possibilearn import *
from tesimodules import Validation
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

def Find_best_model(models, comparison_values, comparison_labels, validation_values=None, validation_labels=None):
    """
    Trova la migliore combinazione di 'c' e 'sigma' basandosi sui risultati del validation_set
    """

    if(validation_values is None or validation_labels is None):
        assert(validation_labels is None and validation_values is None)

    max = .0
    best_model = None
    best_models = {}

    for model in models:
        model_score = models[model].score(comparison_values, comparison_labels)
        if(model_score > max):
            best_model = models[model]
            max = model_score
    
    for model in models:
        model_score = models[model].score(validation_values, validation_labels)
        if(model_score == max):
            best_models[model] = models[model]

    return best_models

def Gen_models(cs, sigmas, classes):
    models = {}
    for c in cs:
        for sigma in sigmas:
            models[c, sigma] = Model_PL(c, sigma)
    
    return models

class Model_PL(BaseEstimator):
    def __init__(self, c=1, sigma=1):
        self.c = c
        self.sigma = sigma

    def fit(self, train_values, train_labels, mu):
        mu_train={}
        indices = []
        labels = []
        classes = list(mu.keys())

        for index, label in train_labels:
            indices.append(index)
            labels.append(label)

        def get_generator(d):
            return lambda m: (-4 + np.random.random(d*m) * 8).reshape((m, d))
            return (-4 + np.random.random(2*m) * 8).reshape((m, 2))

        for clas in classes:
            mu_train[clas] = [mu[clas][i] for i in indices]


        membership_functions = {}
        dimension = len(train_values[0])

        for clas in classes:
            membership_functions[clas], _ = possibility_learn(train_values, 
                                                                mu_train[clas], 
                                                                c=self.c, 
                                                                k=GaussianKernel(self.sigma), 
                                                                sample_generator=get_generator(dimension))

        self.membership_functions = membership_functions
        return self  

    def get_c(self):
        return self.c
    
    def get_sigma(self):
        return self.sigma
    
    def get_hyperparameters(self):
        return (self.c, self.sigma)
    
    def get_membership_function(self, clas):
        return self.membership_functions[clas]

    def __classify(self, item):
        """
        Viene effettuata una predizione di 'item' usando tutte le membership_function e prelevando quella con una probabilità maggiore
        """
        assert(self.membership_functions != None)


        classes = self.membership_functions.keys()
        return sorted([(l, self.get_membership_function(l)(item)) for l in classes], key=lambda i:i[1], reverse=True)[0][0]
    
    def score(self, values, y_true):
        """
        Valuta le prestazioni del modello sul set di 'values' avendo come valori prefissati 'labels'
        """
        assert(len(values) == len(y_true))
        assert(self.membership_functions != None)

        labels = []
        for _, label in y_true:
            labels.append(label)
        print(labels)
        results = list(zip(map(lambda item: self.__classify(item), values), labels))
        validation_result = len([r for r in results if r[0] == r[1]])/len(values)

        return validation_result 
