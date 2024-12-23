# Import Modules
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from Module.utils import load_data, plot_performance

def train_XGB(max_depth, mcw, eta, gamma, sub, bytree, lam, alpha, seed, rounds, feature, batch=1):
    train_loader, test_loader = load_data(feature, batch)

    # Load Training data
    X_train = []; Y_train = []
    for inputs, targets in train_loader:
        X_train.append(inputs.numpy())
        Y_train.append(targets.numpy())
    X_train = np.array(X_train); Y_train = np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = Y_train.reshape(X_train.shape[0], -1)

    # Load Test data
    X_test = []; Y_test = []
    for inputs, targets in test_loader:
        X_test.append(inputs.numpy())
        Y_test.append(targets.numpy())
    X_test = np.array(X_test); Y_test = np.array(Y_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = Y_test.reshape(X_test.shape[0], -1)

    # Generate DMatrix
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': max_depth,
        'min_child_weight': mcw,
        'gamma': gamma,
        'subsample': sub,
        'colsample_bytree': bytree,
        'lambda': lam,
        'alpha': alpha,
        'eta': eta,
        'eval_metric': 'rmse',
        'seed': seed
    }

    evals = [(dtrain, 'train'), (dtest, 'test')]
    bst = xgb.train(params, dtrain, rounds, evals=evals)

    y_pred_train = bst.predict(dtrain)
    y_exact_train = dtrain.get_label()
    print(len(y_exact_train))

    y_pred_test = bst.predict(dtest)
    y_exact_test = dtest.get_label()
    print(len(y_exact_test))

    plot_performance('Test', y_exact_test, y_pred_test)
    plot_performance('Train', y_exact_train, y_pred_train)

def train_RandomForest(n_estimators, max_depth, feature, batch=1):
    train_loader, test_loader = load_data(feature, batch)

    # Load Training data
    X_train = []; Y_train = []
    for inputs, targets in train_loader:
        X_train.append(inputs.numpy())
        Y_train.append(targets.numpy())
    X_train = np.array(X_train); Y_train = np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = Y_train.reshape(X_train.shape[0], -1)

    # Load Test data
    X_test = []; Y_test = []
    for inputs, targets in test_loader:
        X_test.append(inputs.numpy())
        Y_test.append(targets.numpy())
    X_test = np.array(X_test); Y_test = np.array(Y_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = Y_test.reshape(X_test.shape[0], -1)

    reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    reg.fit(X_train, Y_train)

    y_pred_test = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    print(mean_squared_error(Y_test, y_pred_test))

    plot_performance('Test', Y_test, y_pred_test)
    plot_performance('Train', Y_train, y_pred_train)
