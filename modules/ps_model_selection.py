from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from possibilearn import *
import functools

class Validation:

    dataset_values = []
    dataset_labels = []

    def __init__(self, dataset_values, dataset_labels):
        self.dataset_values = dataset_values
        self.dataset_labels = dataset_labels
    
    def get_dataset_values(self):
        return self.dataset_values
    
    def get_dataset_labels(self):
        return self.dataset_labels

    @staticmethod
    def get_best_performance(performances):
        if(len(performances) == 0):
            raise Exception("Performances can't have zero elements") 

        max = -1

        for p in range(len(performances)):
            if performances[p] > max:
                best_index = p
        
        return max

def split_dataset(dataset_values, dataset_labels, random_state=0, size_test=0.2, size_validation=0.2):
    dataset_splitted_test_train = train_test_split(dataset_values, dataset_labels, test_size=size_test) if random_state is None else train_test_split(dataset_values, dataset_labels, test_size=size_test, random_state=random_state) 
    pretrain_values, test_values, pretrain_labels, test_labels = dataset_splitted_test_train

    dataset_splitted_train_validation = train_test_split(pretrain_values, pretrain_labels, test_size=size_validation) if random_state is None else train_test_split(pretrain_values, pretrain_labels, test_size=size_validation, random_state=random_state)
    train_values, validation_values, train_labels, validation_labels = dataset_splitted_train_validation

    return (train_values, train_labels, validation_values, validation_labels, test_values, test_labels)


class Learn:

    def generate_model(self, train_values, train_labels, classes, c=1, sigma=.5):
        """
        Crea un modello composto da una serie di membership_function (una per ogni classe) basate su 'c' e 'sigma'
        Per ogni classe viene generata la membership_function 
        allenata su train_values e train_labels con 'c' e 'sigma' come iperparametri
        """
        membership_functions = {}

        for clas in classes:
            membership_functions[clas], _ = possibility_learn(train_values, train_labels, c=c, k=GaussianKernel(sigma))

        return membership_functions
    
    def generate_models(self, train_values, train_labels, classes, cs, sigmas):
        """
        Genera un modello per ogni combinazione di 'c' e di 'sigma'
        """
        models = {}

        for c in cs:
            for sigma in sigmas:
                models[c, sigma] = generate_model(train_values, train_labels, classes, c, sigma)
        
        return models
    
    def classify(self, item, membership_functions, classes):
        """
        Viene effettuata una predizione di 'item' usando tutte le membership_function e prelevando quella con una probabilità maggiore
        """
        return sorted([(l, membership_functions[l](item)) for l in classes], key=lambda i:i[1], reverse=True)[0][0]

    def find_best_model(self, dataset_values, dataset_labels, classes, cs, sigmas):
        """
        Trova la migliore combinazione di 'c' e 'sigma' basandosi sui risultati del validation_set
        """
        assert(len(cs) > 0)
        assert(len(sigmas) > 0)

        train_values, train_labels, validation_values, validation_labels, _, _ = split_dataset(dataset_values=dataset_values, dataset_labels=dataset_labels)
        mapped_sigmas = map(GaussianKernel, sigmas)

        models = generate_models(train_values, train_labels, classes, cs, mapped_sigmas)

        validation_results = {}

        for model in models:
            key = model.keys()

            results = list(zip(map(lambda item: classify(item, model, classes), validation_values), validation_labels))
            validation_results[key] = len([r for r in results if r[1] == r[2]])/len(validation_values)
        
        return (validation_results)
        # best = Validation.get_best_performance(validation_results)
        # return (model[best])