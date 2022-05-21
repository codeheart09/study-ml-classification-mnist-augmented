from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def lab():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [9, 8, 7, 6, 5, 4]})
    for idx, row in df.iterrows():
        print(row)


def get_dataset():
    print('*** Downloading dataset...')
    return fetch_openml('mnist_784', version=1)


def separate_features_labels(dataset):
    return dataset['data'], dataset['target']


def convert_label_type(labels):
    return labels.astype(np.uint8)


def split_train_test(features, labels):
    return features.iloc[:60000], features.iloc[60000:], labels[:60000], labels[60000:]


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode='constant')
    return shifted_image.reshape([-1])


def augment_train_set(features, labels):
    print('')
    print('*** Augmenting train set...')

    features_np = features.to_numpy()
    labels_np = labels.to_numpy()

    features_augmented = [image for image in features_np]
    labels_augmented = [label for label in labels_np]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for image, label in zip(features_np, labels_np):
            features_augmented.append(shift_image(image, dx, dy))
            labels_augmented.append(label)

    features_augmented = np.array(features_augmented)
    labels_augmented = np.array(labels_augmented)

    shuffle_idx = np.random.permutation(len(features_augmented))
    features_augmented = features_augmented[shuffle_idx]
    labels_augmented = labels_augmented[shuffle_idx]

    return features_augmented, labels_augmented


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
