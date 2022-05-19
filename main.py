from functions import get_dataset, separate_features_labels, convert_label_type, split_train_test, \
    train_model, evaluate_on_test_set

# DATA
dataset = get_dataset()
x, y = separate_features_labels(dataset)
y = convert_label_type(y)
x_train, x_test, y_train, y_test = split_train_test(x, y)
model = train_model(x_train, y_train)
evaluate_on_test_set(model, x_test, y_test)
