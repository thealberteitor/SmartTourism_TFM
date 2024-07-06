import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, auc, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.metrics import geometric_mean_score


RANDOM_STATE = 17
models = {
    'KNN': {
        'model': KNeighborsClassifier(n_neighbors=11, weights='distance'),
        'param_grid': {
            'n_neighbors': [3, 5, 7],
            'weights': ['distance'],
            'p': [1, 2, 3]
        }
    },
    'MLP': {
        'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=30000),
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50,50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05, 0.1],
            'learning_rate': ['constant', 'adaptive'],
        }
    },
    'LR': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, solver='saga', C=0.1),
        'param_grid': {
            'C': [0.1, 1, 10],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'multi_class': ['auto', 'ovr', 'multinomial']
        }
    },
    'SVM': {
        'model': SVC(random_state=RANDOM_STATE, probability=True, kernel='rbf', C=1.0), #poly
        'param_grid': {
            'C': [0.1, 0.5, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    }
}


class TotalModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, random_state):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state

    def perform_grid_search(self, model_name, model, param_grid):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0, scoring='f1_weighted')
        grid_search.fit(self.X_train, self.y_train)
        #print(f"Best parameters found for {model_name}: ", grid_search.best_params_)
        return grid_search.best_estimator_



    def train_evaluate_model(self, model_name, use_grid_search=False):
        if use_grid_search:
            best_model = self.perform_grid_search(model_name, models[model_name]['model'], models[model_name]['param_grid'])
        else:
            best_model = models[model_name]['model']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        y_train_pred = cross_val_predict(best_model, self.X_train, self.y_train, cv=skf, method='predict_proba')
        accuracy_train = accuracy_score(self.y_train, np.argmax(y_train_pred, axis=1))
        train_metrics = precision_recall_fscore_support(self.y_train, np.argmax(y_train_pred, axis=1), zero_division=1, average='weighted')
        g_mean_train = geometric_mean_score(self.y_train, np.argmax(y_train_pred, axis=1), average='weighted')
        train_conf_matrix = confusion_matrix(self.y_train, np.argmax(y_train_pred, axis=1))

        if len(np.unique(self.y_train)) == 2:
            auc_train = roc_auc_score(self.y_train, y_train_pred[:, 1])
        else:
            auc_train = roc_auc_score(self.y_train, y_train_pred, multi_class='ovr')

        best_model.fit(self.X_train, self.y_train)
        self.best_model = best_model

        y_test_pred = best_model.predict(self.X_test)
        y_test_proba = best_model.predict_proba(self.X_test)

        accuracy_test = accuracy_score(self.y_test, y_test_pred)
        test_metrics = precision_recall_fscore_support(self.y_test, y_test_pred, zero_division=1, average='weighted')
        g_mean_test = geometric_mean_score(self.y_test, y_test_pred, average='weighted')
        test_conf_matrix = confusion_matrix(self.y_test, y_test_pred)

        #AUC
        if len(np.unique(self.y_test)) == 2:
            auc_test = roc_auc_score(self.y_test, y_test_proba[:, 1])
        else:
            auc_test = roc_auc_score(self.y_test, y_test_proba, multi_class='ovr')

        print("Testing Metrics:")
        #print(f"Precision: {test_metrics[0]:.4f}, Recall: {test_metrics[1]:.4f}, F1-score: {test_metrics[2]:.4f}, G-Mean: {g_mean_test:.4f}, AUC: {auc_test:.4f}")
        print(f"F1-score: {test_metrics[2]:.4f}, G-Mean: {g_mean_test:.4f}, Recall: {test_metrics[1]:.4f}, AUC: {auc_test:.4f}")

        #Print confusion matrix
        #print("Confusion Matrix:\n", test_conf_matrix)
        return best_model, []
