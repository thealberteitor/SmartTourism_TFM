import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from KMedoidsSMOTE import KMedoidsSMOTE
from data_preprocessor import DataPreprocessor
from total_model_trainer import models, TotalModelTrainer
import smote_variants as sv

RANDOM_STATE = 17
TEST_SIZE = 0.2
GRID_SEARCH = False
DO_SMOTE = True

def main():
    file_path = "../../DB/surveys/v2modified/smart_poqueira.csv"
    dataframe = pd.read_csv(file_path)

    columnas_seleccionadas_vehicles = [
        'avg_gross_income', 'population', 'km_to_POQ', 'avg_nights', 'std_nights',
        'avg_holiday', 'std_holiday', 'avg_workday', 'std_workday', 'avg_high_season',
        'std_high_season', 'avg_low_season', 'std_low_season', 'total_distance',
        'total_holiday', 'total_workday', 'total_high_season', 'total_low_season',
        'entry_in_high_season', 'entry_in_holiday', 'nights', 'fidelity',
        'visits_dif_weeks', 'visits_dif_months', 'total_entries',
        'visit_time', 'distance', 'visits_dif_weeks', 'visits_dif_months', 'fidelity',
        'cumulative_entries', 'num_holiday', 'num_workday', 'num_high_season',
        'num_low_season', 'entry_in_holiday', 'entry_in_high_season', 'km_to_dest',
        'num_seats', 'environmental_distinctive', 'co2_emissions'
    ]

    preprocessor = DataPreprocessor(columnas_seleccionadas_vehicles, dataframe, 'label', RANDOM_STATE) #label = likelihood
    X_train, X_test, y_train, y_test = train_test_split(preprocessor.X_processed, preprocessor.y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))

    print("Number of samples in each class in the training set:")
    print(class_counts)

    X_train = X_train.values
    X_test = X_test.values

    if DO_SMOTE:
        smote_start_time = time.time()
        smote = KMedoidsSMOTE(sampling_strategy='not majority', random_state=RANDOM_STATE, metric="manhattan")
        #smote = SMOTE(sampling_strategy='not majority', random_state=RANDOM_STATE)
        #smote = KMeansSMOTE(sampling_strategy='not majority', random_state=RANDOM_STATE, cluster_balance_threshold=0.01)
        #smote = sv.CURE_SMOTE(random_state=RANDOM_STATE)

        X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
        smote_duration = time.time() - smote_start_time
    else:
        X_train_oversampled, y_train_oversampled = X_train, y_train

    y_test_binarized = label_binarize(y_test, classes=np.unique(y_train_oversampled))
    plt.figure(figsize=(10, 8))
    for model_name in models.keys():
        print(f"\n{model_name}")
        trainer = TotalModelTrainer(X_train_oversampled, y_train_oversampled, X_test, y_test, RANDOM_STATE)
        model_start_time = time.time()
        best_model, _ = trainer.train_evaluate_model(model_name, use_grid_search=GRID_SEARCH)
        model_duration = time.time() - model_start_time + smote_duration
        print(f"Total duration for {model_name}: {model_duration:.2f} seconds")

        y_score = best_model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
