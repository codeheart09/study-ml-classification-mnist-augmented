from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.ndimage.interpolation import shift
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


def augment_train_set(features, labels):
    shifted_features = []
    shifted_labels = []

    for row in features.itertuples():
        image = np.array(list(row)).reshape(28, 28)
        shifted1 = shift(image, [1, 0], cval=0)
        shifted2 = shift(image, [-1, 0], cval=0)
        shifted3 = shift(image, [0, 1], cval=0)
        shifted4 = shift(image, [0, -1], cval=0)
        shifted_features.append(shifted1.flatten())
        shifted_features.append(shifted2.flatten())
        shifted_features.append(shifted3.flatten())
        shifted_features.append(shifted4.flatten())
        shifted_labels.append(labels[row.Index])
        shifted_labels.append(labels[row.Index])
        shifted_labels.append(labels[row.Index])
        shifted_labels.append(labels[row.Index])

    for row in shifted_features:
        # @todo continue here




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
