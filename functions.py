from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def get_dataset():
    print('*** Downloading dataset...')
    return fetch_openml('mnist_784', version=1)


def separate_features_labels(dataset):
    return dataset['data'], dataset['target']


def convert_label_type(labels):
    return labels.astype(np.uint8)


def split_train_test(features, labels):
    return features.iloc[:60000], features.iloc[60000:], labels[:60000], labels[60000:]


def train_model(features, labels):
    print('')
    print('*** Training model...')
    model = KNeighborsClassifier()

    param_grid = [
        {'weights': ['uniform', 'distance'], 'n_neighbors': [1, 3, 5, 7, 9]}
    ]
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
    )
    grid_search.fit(features, labels)

    print('Grid Search best prams:', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    best_model.fit(features, labels)
    return best_model


def evaluate_on_test_set(model, features, labels):
    print('')
    print('*** Evaluating on test set...')
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f'Model accuracy: {accuracy}')
