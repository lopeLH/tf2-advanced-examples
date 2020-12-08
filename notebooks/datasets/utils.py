import numpy as np

def generate_train_and_test_data(sample_generation_function,
                                 n_samples_train=10000, n_samples_test=1000, **kargs):

    X_train, Y_train, X_test, Y_test = [], [], [], []

    for _ in range(n_samples_train):
        img, mask = sample_generation_function(**kargs)
        X_train.append(img)
        Y_train.append(mask)

    for _ in range(n_samples_test):
        img, mask = sample_generation_function(**kargs)
        X_test.append(img)
        Y_test.append(mask)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)

    return (X_train, Y_train), (X_test, Y_test)
