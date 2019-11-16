import numpy as np

def __get_indices(dataset_values, dataset_labels, indices):
    return ([dataset_values[i] for i in indices], [dataset_labels[i] for i in indices])

def split_train_validation_test_dataset(dataset_values, dataset_labels, random_state=None, size_train=0.6, size_validation=0.2):
    assert(len(dataset_values) == len(dataset_labels))
    assert(0 < size_train + size_validation < 1)

    n = len(dataset_labels) # ..or dataset_values

    permutation_indices = np.random.permutation(len(dataset_labels)) if random_state is None else np.random.RandomState(seed=random_state).permutation(len(dataset_labels))

    train_lenght = int(n * size_train)
    train_with_validation_length = int(n * (size_train + size_validation))

    train_indices = permutation_indices[:train_lenght]
    validation_indices = permutation_indices[train_lenght : train_with_validation_length]
    test_indices = permutation_indices[train_with_validation_length:]

    train_values, train_labels = __get_indices(dataset_values, dataset_labels, train_indices)
    validation_values, validation_labels = __get_indices(dataset_values, dataset_labels, validation_indices)
    test_values, test_labels = __get_indices(dataset_values, dataset_labels, test_indices)

    return (train_values, train_labels, validation_values, validation_labels, test_values, test_labels, train_indices, validation_indices, test_indices)

def split_train_test_dataset(dataset_values, dataset_labels, random_state=None, size_train=0.6):
    assert(len(dataset_values) == len(dataset_labels))
    assert(0 < size_train < 1)

    n = len(dataset_labels)

    permutation_indices = np.random.permutation(len(dataset_labels)) if random_state is None else np.random.RandomState(seed=random_state).permutation(len(dataset_labels))

    train_length = int(n * size_train)

    train_indices = permutation_indices[:train_length]
    test_indices = permutation_indices[train_length:]

    train_values, train_labels = __get_indices(dataset_values, dataset_labels, train_indices)
    test_values, test_labels = __get_indices(dataset_values, dataset_labels, test_indices)

    return (train_values, train_labels, test_values, test_labels, train_indices, test_indices)