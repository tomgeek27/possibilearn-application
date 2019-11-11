from possibilearn import *
from tesimodules import Validation

class Model_PL:
    def __init__(self, c, sigma, membership_functions):
        self.c = c
        self.sigma = sigma

        self.membership_functions = membership_functions
    
    def get_c(self):
        return self.c
    
    def get_sigma(self):
        return self.sigma
    
    def get_membership_function(self, clas):
        return self.membership_functions[clas]

    def __classify(self, item):
        """
        Viene effettuata una predizione di 'item' usando tutte le membership_function e prelevando quella con una probabilità maggiore
        """
        classes = self.membership_functions.keys()
        return sorted([(l, self.get_membership_function(l)(item)) for l in classes], key=lambda i:i[1], reverse=True)[0][0]
    
    def score(self, values, labels):
        """
        Valuta le prestazioni del modello sul set di 'values' avendo come valori prefissati 'labels'
        """
        assert(len(values) == len(labels))

        results = list(zip(map(lambda item: self.__classify(item), values), labels))
        validation_result = len([r for r in results if r[0] == r[1]])/len(values)

        return validation_result 


class Learn:

    def __init__(self, dataset_values, dataset_labels, classes, random_state=None, size_train=0.6, size_validation=0.2):
        self.dataset_values = dataset_values
        self.dataset_labels = dataset_labels
        self.classes = classes

        self.generate_train_validation_test_sets(random_state=random_state, size_train=size_train, size_validation=size_validation)

    def __generate_mu(self):
        mu = {}

        def to_membership_values(labels, target):
            return [1 if l==target else 0 for l in labels]

        for clas in self.classes:
            mu[clas] = to_membership_values(self.dataset_labels, clas)

        return mu
    
    def __generate_specific_mu(self, indices):
        mu_general = {}
        for clas in self.classes:
            mu_general[clas] = [self.mu[clas][i] for i in indices]

        return mu_general

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

    def get_values_sets(self):
        return (self.get_values_train(), self.get_values_validation(), self.get_values_test())

    def get_labels_sets(self):
        return (self.get_labels_train(), self.get_labels_validation(), self.get_labels_test())

    def generate_train_validation_test_sets(self, random_state, size_train, size_validation):
        self.train_values, self.train_labels, self.validation_values, self.validation_labels, self.test_values, self.test_labels, self.train_indices, self.validation_indices, self.test_indices = Validation.split_dataset(dataset_values=self.dataset_values, dataset_labels=self.dataset_labels, random_state=random_state, size_train=size_train, size_validation=size_validation)
        self.mu = self.__generate_mu()
        self.mu_train = self.__generate_specific_mu(self.train_indices)

    def generate_model(self, c, sigma):

        membership_functions = {}
        
        def get_generator(d):
            return lambda m: (-4 + np.random.random(d*m) * 8).reshape((m, d))
            return (-4 + np.random.random(2*m) * 8).reshape((m, 2))
    
        for clas in self.classes:
            membership_functions[clas], _ = possibility_learn(self.train_values, self.mu_train[clas], c=c, k=GaussianKernel(sigma), sample_generator=get_generator(4))

        return Model_PL(c, sigma, membership_functions)

    def generate_models(self, cs, sigmas):
        """
        Genera un modello per ogni combinazione di 'c' e di 'sigma'
        """
        models = {}

        for c in cs:
            for sigma in sigmas:
                models[c, sigma] = self.generate_model(c, sigma)

        return models

    def find_best_model(self, models):
        """
        Trova la migliore combinazione di 'c' e 'sigma' basandosi sui risultati del validation_set
        """

        max = 0
        best_model = None

        for model in models:
            model_score = models[model].score(self.validation_values, self.validation_labels)
            if(model_score > max):
                best_model = models[model]
                max = model_score
        
        return best_model