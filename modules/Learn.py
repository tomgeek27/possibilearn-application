from possibilearn import *
from tesimodules import Validation

def generate_model(train_values, dataset_labels, train_indices, classes, c=1, sigma=.5):
    """
    Crea un modello composto da una serie di membership_function (una per ogni classe) basate su 'c' e 'sigma'
    Per ogni classe viene generata la membership_function 
    allenata su train_values e train_labels con 'c' e 'sigma' come iperparametri
    """        
    membership_functions = {}
    mu = {}
    mu_train = {}

    def to_membership_values(labels, target):
        return [1 if l==target else 0 for l in labels]

    def get_generator(d):
        return lambda m: (-4 + np.random.random(d*m) * 8).reshape((m, d))
        return (-4 + np.random.random(2*m) * 8).reshape((m, 2))

    for clas in classes:
        mu[clas] = to_membership_values(dataset_labels, clas)
        mu_train[clas] = [mu[clas][i] for i in train_indices]

        membership_functions[clas], _ = possibility_learn(train_values, mu_train[clas], c=c, k=GaussianKernel(sigma), sample_generator=get_generator(4))

    return membership_functions

def generate_models(train_values, dataset_labels, train_indices, classes, cs, sigmas):
    """
    Genera un modello per ogni combinazione di 'c' e di 'sigma'
    """
    models = {}

    for c in cs:
        for sigma in sigmas:
            models[c, sigma] = generate_model(train_values, dataset_labels, train_indices, classes, c, sigma)

    return models

def classify(item, membership_functions):#, classes):
    """
    Viene effettuata una predizione di 'item' usando tutte le membership_function e prelevando quella con una probabilità maggiore
    """
    classes = membership_functions.keys()
    return sorted([(l, membership_functions[l](item)) for l in classes], key=lambda i:i[1], reverse=True)[0][0]

def get_performance(values, labels, model):
    assert(len(values) == len(labels))

    results = list(zip(map(lambda item: classify(item, model), values), labels))
    validation_result = len([r for r in results if r[0] == r[1]])/len(values)

    return validation_result   

def find_best_model(dataset_values, dataset_labels, classes, cs, sigmas):
    """
    Trova la migliore combinazione di 'c' e 'sigma' basandosi sui risultati del validation_set
    """
    assert(len(cs) > 0)
    assert(len(sigmas) > 0)

    train_values, train_labels, validation_values, validation_labels, _, _, train_indices, validation_indices, _ = Validation.split_dataset(dataset_values=dataset_values, dataset_labels=dataset_labels)

    mapped_sigmas = map(GaussianKernel, sigmas)

    models = generate_models(train_values, dataset_labels, train_indices, classes, cs, sigmas)

    validation_results = {}

    for model_key in models.keys():
        model = models[model_key]
        validation_results[model_key] = get_performance(validation_values, validation_labels, model)
    
    print(validation_results)
    best = Validation.get_best_performance(validation_results)
    return (models[best])   
