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
        assert(len(values) == len(labels))

        results = list(zip(map(lambda item: self.__classify(item), values), labels))
        validation_result = len([r for r in results if r[0] == r[1]])/len(values)

        return validation_result 


class Learn:

    def generate_train_validation_test_sets(self, random_state, size_train, size_validation):
        self.train_values, self.train_labels, self.validation_values, self.validation_labels, self.test_values, self.test_labels, self.train_indices, self.validation_indices, self.test_indices = Validation.split_dataset(dataset_values=self.dataset_values, dataset_labels=self.dataset_labels, random_state=random_state, size_train=size_train, size_validation=size_validation)
        self.mu = self.__generate_mu()
        self.mu_train = self.__generate_mu_general(self.train_indices)

    def __init__(self, dataset_values, dataset_labels, classes, random_state=None, size_train=0.6, size_validation=0.2):
        self.dataset_values = dataset_values
        self.dataset_labels = dataset_labels
        self.classes = classes

        self.generate_train_validation_test_sets(random_state=random_state, size_train=size_train, size_validation=size_validation)

    def get_values_sets(self):
        return (self.train_values, self.validation_values, self.test_values)

    def get_labels_sets(self):
        return (self.train_labels, self.validation_labels, self.test_labels)

    def __generate_mu(self):
        mu = {}

        def to_membership_values(labels, target):
            return [1 if l==target else 0 for l in labels]

        for clas in self.classes:
            mu[clas] = to_membership_values(self.dataset_labels, clas)

        return mu
    
    def __generate_mu_general(self, indices):
        mu_general = {}
        for clas in self.classes:
            mu_general[clas] = [self.mu[clas][i] for i in indices]

        return mu_general

    # def __classify(self, item, membership_functions):
    #     """
    #     Viene effettuata una predizione di 'item' usando tutte le membership_function e prelevando quella con una probabilità maggiore
    #     """
    #     classes = membership_functions.keys()
    #     return sorted([(l, membership_functions[l](item)) for l in classes], key=lambda i:i[1], reverse=True)[0][0]
    
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

        #mapped_sigmas = map(kernel, )

        #models = generate_models(train_values, dataset_labels, train_indices, classes, cs, sigmas)

        # validation_results = {}

        max = 0
        best_model = None

        for model in models:
            model_score = model.score(self.validation_values, self.validation_labels)
            if(model_score > max):
                best_model = model
                max = model_score
        
        return (model)

        # for model_key in models.keys():
        #     model = models[model_key]
        #     validation_results[model_key] = get_performance(validation_values, validation_labels, model)
        
        # print(validation_results)
        # best = Validation.get_best_performance(validation_results)
        # return (models[best]) 
    # def get_performance(self, values, labels, model):

        

    #     assert(len(values) == len(labels))

    #     results = list(zip(map(lambda item: __classify(item, model), values), labels))
    #     validation_result = len([r for r in results if r[0] == r[1]])/len(values)

    #     return validation_result 


# def generate_model(train_values, dataset_labels, train_indices, classes, c=1, sigma=.5):
#     """
#     Crea un modello composto da una serie di membership_function (una per ogni classe) basate su 'c' e 'sigma'
#     """        
#     membership_functions = {}
#     mu = generate_mu(dataset_labels, classes)
#     mu_train = {}

#     def to_membership_values(labels, target):
#         return [1 if l==target else 0 for l in labels]

#     def get_generator(d):
#         return lambda m: (-4 + np.random.random(d*m) * 8).reshape((m, d))
#         return (-4 + np.random.random(2*m) * 8).reshape((m, 2))

#     for clas in classes:
#         mu_train[clas] = [mu[clas][i] for i in train_indices]

#         membership_functions[clas], _ = possibility_learn(train_values, mu_train[clas], c=c, k=GaussianKernel(sigma), sample_generator=get_generator(4))

#     return membership_functions

# def generate_models(train_values, dataset_labels, train_indices, classes, cs, sigmas):
#     """
#     Genera un modello per ogni combinazione di 'c' e di 'sigma'
#     """
#     models = {}

#     mu = generate_mu(dataset_labels, classes)
#     mu_train = 

#     for c in cs:
#         for sigma in sigmas:
#             models[c, sigma] = generate_model(train_values, dataset_labels, train_indices, classes, c, sigma)

#     return models

# def classify(item, membership_functions):#, classes):
#     """
#     Viene effettuata una predizione di 'item' usando tutte le membership_function e prelevando quella con una probabilità maggiore
#     """
#     classes = membership_functions.keys()
#     return sorted([(l, membership_functions[l](item)) for l in classes], key=lambda i:i[1], reverse=True)[0][0]

# def get_performance(values, labels, model):
#     assert(len(values) == len(labels))

#     results = list(zip(map(lambda item: classify(item, model), values), labels))
#     validation_result = len([r for r in results if r[0] == r[1]])/len(values)

#     return validation_result   

# def find_best_model(dataset_values, dataset_labels, classes, cs, sigmas):
#     """
#     Trova la migliore combinazione di 'c' e 'sigma' basandosi sui risultati del validation_set
#     """
#     assert(len(cs) > 0)
#     assert(len(sigmas) > 0)

#     train_values, train_labels, validation_values, validation_labels, _, _, train_indices, validation_indices, _ = Validation.split_dataset(dataset_values=dataset_values, dataset_labels=dataset_labels)

#     mapped_sigmas = map(GaussianKernel, sigmas)

#     models = generate_models(train_values, dataset_labels, train_indices, classes, cs, sigmas)

#     validation_results = {}

#     for model_key in models.keys():
#         model = models[model_key]
#         validation_results[model_key] = get_performance(validation_values, validation_labels, model)
    
#     print(validation_results)
#     best = Validation.get_best_performance(validation_results)
#     return (models[best])   
