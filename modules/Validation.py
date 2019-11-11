from sklearn.model_selection import train_test_split
import numpy as np

def get_best_performance(performances):
    if(len(performances) == 0):
        raise Exception("Performances can't have zero elements") 

    if (type(performances) != dict):
        raise Exception("Performances must be a dict with hyperparamters as key")

    max = -1
    best_index = list(performances.keys())[0]

    for p in performances.keys():
        if performances[p] > max:
            best_index = p
            max = performances[best_index]
    
    return best_index

def split_dataset(dataset_values, dataset_labels, random_state=None, size_train=0.6, size_validation=0.2):
    assert(len(dataset_values) == len(dataset_labels))
    assert(0 < size_train + size_validation < 1)

    n = len(dataset_labels) # ..or dataset_values

    permutation_indices = np.random.permutation(len(dataset_labels)) if random_state is None else np.random.RandomState(seed=random_state).permutation(len(dataset_labels))

    train_lenght = int(n * size_train)
    train_with_validation_length = int(n * (size_train + size_validation))

    train_indices = permutation_indices[:train_lenght]
    validation_indices = permutation_indices[train_lenght : train_with_validation_length]
    test_indices = permutation_indices[train_with_validation_length:]

    train_values = [dataset_values[i] for i in train_indices]
    train_labels = [dataset_labels[i] for i in train_indices]

    validation_values = [dataset_values[i] for i in validation_indices]
    validation_labels = [dataset_labels[i] for i in validation_indices]

    test_values = [dataset_values[i] for i in test_indices]
    test_labels = [dataset_labels[i] for i in test_indices]

    # dataset_splitted_test_train = train_test_split(dataset_values, dataset_labels, indices, test_size=size_test) if random_state is None else train_test_split(dataset_values, dataset_labels, indices, test_size=size_test, random_state=random_state) 
    # pretrain_values, test_values, pretrain_labels, test_labels, pretrain_indices, test_indices = dataset_splitted_test_train

    # dataset_splitted_train_validation = train_test_split(pretrain_values, pretrain_labels, indices, test_size=size_validation) if random_state is None else train_test_split(pretrain_values, pretrain_labels, indices, test_size=size_validation, random_state=random_state)
    # train_values, validation_values, train_labels, validation_labels, train_indices, validation_indices = dataset_splitted_train_validation

    return (train_values, train_labels, validation_values, validation_labels, test_values, test_labels, train_indices, validation_indices, test_indices)